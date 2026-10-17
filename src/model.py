"""A small GPT over commaVQ tokens.

Each frame is 8x16 = 128 VQ tokens (vocab 1024). A sequence is
  [BOS, f0_0 .. f0_127, BOS, f1_0 .. f1_127, ...]
so every frame starts with BOS (id 1024) and the model predicts frames in raster order,
each token conditioned on all earlier frames plus the earlier tokens of its own frame.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 1024
BOS = 1024
TOK_PER_FRAME = 128


@dataclass
class Config:
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    frames: int = 16          # context length in frames
    dropout: float = 0.0
    action_dim: int = 0       # >0: condition each frame on the ego motion leading into it (M4)
    act_mean: tuple = ()      # normalisation for the actions, fit on train
    act_std: tuple = ()

    @property
    def block(self) -> int:
        return self.frames * (TOK_PER_FRAME + 1)


class Block(nn.Module):
    """Pre-norm transformer block with flash attention (causal SDPA)."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_head = c.n_head
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model, bias=False), nn.GELU(),
                                 nn.Linear(4 * c.d_model, c.d_model, bias=False))
        self.drop = c.dropout

    def forward(self, x):
        """x (B,T,D) -> (B,T,D)."""
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_head, D // self.n_head).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class WorldModel(nn.Module):
    """Token GPT with separate learned embeddings for (frame index, position within frame)."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(VOCAB + 1, c.d_model)
        self.pos_in_frame = nn.Embedding(TOK_PER_FRAME + 1, c.d_model)
        self.frame_idx = nn.Embedding(c.frames, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, VOCAB, bias=False)   # never predicts BOS
        self.apply(self._init)
        if c.action_dim:
            # built after init and zero-initialised: same seed -> identical weights to the base model and the
            # same function at step 0, so any difference after training comes from the pose alone
            self.act = nn.Linear(c.action_dim, c.d_model)
            nn.init.zeros_(self.act.weight); nn.init.zeros_(self.act.bias)
            self.register_buffer("act_mean", torch.tensor(c.act_mean, dtype=torch.float32))
            self.register_buffer("act_std", torch.tensor(c.act_std, dtype=torch.float32))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, act=None):
        """idx (B,T) token ids incl. BOS -> logits (B,T,VOCAB) for the *next* token.
        act (B,>=F-1,6): ego motion leading into frames 1.., added to every token (and BOS) of that
        frame, so frame t's tokens see their own action but earlier frames never do."""
        B, T = idx.shape
        t = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos_in_frame(t % (TOK_PER_FRAME + 1)) + self.frame_idx(t // (TOK_PER_FRAME + 1))
        if self.c.action_dim:
            assert act is not None, "this model needs actions"
            nf = (T + TOK_PER_FRAME) // (TOK_PER_FRAME + 1)                # frames touched by idx
            e = self.act(((act[:, :nf - 1] - self.act_mean) / self.act_std).to(x.dtype))
            e = torch.cat([torch.zeros_like(e[:, :1]), e], 1)              # frame 0 has no incoming action
            x = x + e.repeat_interleave(TOK_PER_FRAME + 1, 1)[:, :T]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate_frame(self, ctx_frames: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, act=None):
        """ctx_frames (B,F,128) with F < c.frames -> next frame (B,128). Sliding-window, no KV cache:
        simple and fast enough for a ~50M model rendering a few seconds of video.
        act (B,F,6): actions into frames 1..F of the context plus the one into the new frame."""
        B, Fr, _ = ctx_frames.shape
        seq = to_sequence(ctx_frames)
        seq = torch.cat([seq, torch.full((B, 1), BOS, device=seq.device, dtype=seq.dtype)], 1)
        for _ in range(TOK_PER_FRAME):
            logits = self(seq, act)[:, -1].float()
            if temperature == 0:                                     # greedy; dividing by 0 would give inf/nan
                seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], 1)
                continue
            logits = logits / temperature
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = -float("inf")
            nxt = torch.multinomial(logits.softmax(-1), 1)
            seq = torch.cat([seq, nxt], 1)
        return seq[:, -TOK_PER_FRAME:]


def to_sequence(frames: torch.Tensor) -> torch.Tensor:
    """(B,F,128) frame tokens -> (B, F*129) with a BOS before every frame."""
    B, Fr, _ = frames.shape
    bos = torch.full((B, Fr, 1), BOS, device=frames.device, dtype=frames.dtype)
    return torch.cat([bos, frames], 2).reshape(B, Fr * (TOK_PER_FRAME + 1))


def frame_loss(model: WorldModel, frames: torch.Tensor, act=None, skip_frames: int = 1):
    """Mean next-token cross-entropy (nats) over the tokens of frames[skip_frames:].
    Frame 0 has no past, so including it would mix 'predict the next frame' with
    'guess a frame from nothing'; we score only frames that have at least one frame of context."""
    seq = to_sequence(frames)
    logits = model(seq[:, :-1]) if act is None else model(seq[:, :-1], act)
    tgt = seq[:, 1:].clone()
    tgt[tgt == BOS] = -100                                      # never score predicting BOS
    tgt[:, : skip_frames * (TOK_PER_FRAME + 1) - 1] = -100
    return F.cross_entropy(logits.float().reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=-100)
