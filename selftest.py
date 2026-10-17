"""Checks that need no real data. Run: make selftest

What can go silently wrong in a token world model: a leaky causal mask (the model "predicts"
tokens it can already see, loss looks great), scoring frame 0 or BOS tokens, a sampler that
reads across segment boundaries, and a converter that scrambles frame order.
"""
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.data import FRAMES_PER_SEG, Windows, load_tokens, prepare_shard
from src.model import BOS, TOK_PER_FRAME, Config, WorldModel, frame_loss, to_sequence

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# --- sequence layout --------------------------------------------------------
fr = torch.arange(3 * 128).view(1, 3, 128) % 1024
seq = to_sequence(fr)
check("to_sequence: BOS before each frame, 3*129 long",
      seq.shape == (1, 387) and (seq[0, [0, 129, 258]] == BOS).all() and seq[0, 130].item() == fr[0, 1, 0].item())

# --- causality: changing token j must not change logits at positions < j ----
cfg = Config(n_layer=2, n_head=4, d_model=64, frames=4)
m = WorldModel(cfg).to(dev).eval()
x = to_sequence(torch.randint(0, 1024, (1, 4, 128), device=dev))
j = 300
y = x.clone(); y[0, j] = (y[0, j] + 1) % 1024
with torch.no_grad():
    a, b = m(x), m(y)
check("causal: editing token 300 leaves logits 0..299 unchanged", torch.allclose(a[0, :j], b[0, :j], atol=1e-5))
check("causal: ...and does change logits at 300+", not torch.allclose(a[0, j:], b[0, j:]))

# --- action conditioning (M4): frame t's action reaches frame t's tokens, never earlier ones
ca = Config(n_layer=2, n_head=4, d_model=64, frames=4, action_dim=6, act_mean=(0.0,) * 6, act_std=(1.0,) * 6)
ma = WorldModel(ca).to(dev).eval()
torch.manual_seed(1); mb = WorldModel(Config(n_layer=2, n_head=4, d_model=64, frames=4)).to(dev).eval()
torch.manual_seed(1); mz = WorldModel(ca).to(dev).eval()
with torch.no_grad():
    same = torch.equal(mb(x), mz(x, torch.randn(1, 3, 6, device=dev)))
check("pose model at init = base model (same seed, zero-init action layer)", same)
torch.nn.init.normal_(ma.act.weight, std=0.02)                              # zero-init would make the next checks vacuous
act = torch.randn(1, 3, 6, device=dev)                                       # actions into frames 1,2,3
act2 = act.clone(); act2[0, 1] += 1.0                                         # change the action into frame 2
with torch.no_grad():
    a, b = ma(x, act), ma(x, act2)
start2 = 2 * 129                                                              # BOS of frame 2
check("action into frame 2 leaves logits before frame 2's BOS unchanged",
      torch.allclose(a[0, :start2], b[0, :start2], atol=1e-5))
check("...and changes the predictions of frame 2's tokens", not torch.allclose(a[0, start2:start2 + 129], b[0, start2:start2 + 129]))
with torch.no_grad():
    g1 = ma.generate_frame(x.view(1, 4, 129)[:, :3, 1:], temperature=0, act=act)
check("generate_frame runs with actions", g1.shape == (1, 128))

# --- loss masks frame 0 and BOS targets ------------------------------------
f = torch.randint(0, 1024, (2, 4, 128), device=dev)
g = f.clone(); g[:, 0] = torch.randint(0, 1024, (2, 128), device=dev)      # change frame 0 only
with torch.no_grad():
    l1, l2 = frame_loss(m, f), frame_loss(m, g)
check("frame_loss responds to frame 0 only as context (value changes)", abs(l1.item() - l2.item()) > 1e-6)

class Peek(torch.nn.Module):
    """Fake model: puts all mass on the true next token, except at BOS-target and frame-0 positions."""
    def __init__(self, seq): super().__init__(); self.seq = seq
    def forward(self, s):
        logits = torch.full((*s.shape, 1024), -30.0, device=s.device)
        nxt = self.seq[:, 1:]
        ok = nxt < 1024
        logits[ok.nonzero(as_tuple=True) + (nxt[ok],)] = 30.0
        return logits
s = to_sequence(f)
check("frame_loss scores exactly the tokens of frames 1..F-1 (perfect predictor -> ~0)",
      frame_loss(Peek(s), f).item() < 1e-6)
bad = Peek(s); bad.seq = s.clone(); bad.seq[:, 1:129] = 0                  # wrong on frame 0 only
check("...and ignores frame-0 targets (wrong on frame 0 still ~0)", frame_loss(bad, f).item() < 1e-6)
bad2 = Peek(s); bad2.seq = s.clone(); bad2.seq[:, 130] = (s[:, 130] + 1) % 1024
check("...but a single miss in frame 1 is counted", frame_loss(bad2, f).item() > 0.1)

# --- converter round-trip on a synthetic shard ------------------------------
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    rng = np.random.default_rng(0)
    segs = {f"seg{i}": (rng.integers(0, 1024, (FRAMES_PER_SEG, 8, 16)).astype(np.int16),
                        rng.standard_normal((FRAMES_PER_SEG, 6)).astype(np.float32)) for i in range(3)}
    with tarfile.open(tmp / "data-9999.tar.gz", "w:gz") as tar:
        for name in ["seg0", "seg1", "seg2"]:
            for kind, arr in zip(["token", "pose"], segs[name]):
                buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
                ti = tarfile.TarInfo(f"{name}.{kind}.npy"); ti.size = len(buf.getbuffer())
                tar.addfile(ti, buf)
            meta = json.dumps({"file_name": name}).encode()
            ti = tarfile.TarInfo(f"{name}.json"); ti.size = len(meta); tar.addfile(ti, io.BytesIO(meta))
    idx = prepare_shard(tmp / "data-9999.tar.gz", tmp / "tok")
    T = load_tokens(tmp / "tok", ["data-9999"])
    order = idx["segments"]
    check("prepare_shard keeps every segment", idx["n"] == 3 and sorted(order) == ["seg0", "seg1", "seg2"])
    check("tokens round-trip exactly, frame order intact",
          all(np.array_equal(T[i], segs[n][0].reshape(FRAMES_PER_SEG, 128)) for i, n in enumerate(order)))
    P = np.memmap(tmp / "tok" / "data-9999.pose.bin", np.float32, "r", shape=(3, FRAMES_PER_SEG, 6))
    check("poses round-trip exactly", all(np.array_equal(P[i], segs[n][1]) for i, n in enumerate(order)))

    # --- sampler never crosses a segment boundary, honours stride ----------
    w = Windows(T, frames=16, stride=2, seed=0)
    bt = w.batch(64, device="cpu").numpy()
    ok = 0
    for row in bt:
        hits = [(i, a) for i in range(3) for a in range(FRAMES_PER_SEG - 30)
                if np.array_equal(T[i, a], row[0]) and np.array_equal(T[i, a + 30], row[15])]
        ok += bool(hits) and all(np.array_equal(T[hits[0][0], hits[0][1] + 2 * k], row[k]) for k in range(16))
    check("Windows: every sample = 16 frames, stride 2, inside one segment", ok == 64, f"{ok}/64")
    check("Windows.fixed is deterministic", w.fixed(5) == w.fixed(5))
    wp = Windows(T, frames=16, seed=0, pose=P)
    pairs = wp.fixed(8)
    xt, at = wp.take(pairs, device="cpu")
    check("Windows with pose: actions = pose[start+i-1] for frames i=1..15",
          at.shape == (8, 15, 6) and all(np.array_equal(at[k].numpy(), P[s, st:st + 15]) for k, (s, st) in enumerate(pairs))
          and all(np.array_equal(xt[k].numpy(), T[s, st:st + 16]) for k, (s, st) in enumerate(pairs)))

# --- the real model learns: overfit a tiny repeated clip ---------------------
cfg = Config(n_layer=4, n_head=4, d_model=128, frames=4)
m = WorldModel(cfg).to(dev)
clip = torch.randint(0, 1024, (1, 128), device=dev)
moving = torch.stack([torch.roll(clip, k, 1) for k in range(4)], 1).expand(8, 4, 128)  # each frame = last shifted by 1
opt = torch.optim.AdamW(m.parameters(), 1e-3)
for step in range(400):
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        loss = frame_loss(m, moving)
    opt.zero_grad(); loss.backward(); opt.step()
    if step == 0: first = loss.item()
check("model learns a shifting clip (loss 6.9 -> < 0.1)", first > 6 and loss.item() < 0.1, f"{first:.2f} -> {loss.item():.4f}")
m.eval()
nxt = m.generate_frame(moving[:1, :3], temperature=0)
check("greedy generate_frame reproduces the next shifted frame", torch.equal(nxt[0], moving[0, 3]),
      f"{(nxt[0] == moving[0, 3]).float().mean().item():.2%} tokens right")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
