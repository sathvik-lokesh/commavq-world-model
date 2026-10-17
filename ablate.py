"""M2: does context help, or is the model a smart copy machine? Run: make ablate

Same model, same target frames, different amounts of past. For each fixed test window of 16
frames the target is the last frame; with k frames of context the model sees frames[15-k:16]
and only the target's 128 tokens are scored. Frame k always saw exactly k past frames during
training, so every k here is in-distribution (no frame-index trick).

Tokens are split into "repeat" (same id as the same position in the previous frame) and
"changed". A copier gets the repeat tokens cheaply from 1 frame of context; if extra context
only helps there, the model isn't learning motion. The prev-token baseline is scored on the
very same target tokens.

The windows use a different seed from the training eval windows, which best.pt was selected on.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import Windows, load_tokens
from src.model import BOS, TOK_PER_FRAME, VOCAB, Config, WorldModel, to_sequence


@torch.no_grad()
def target_nll(model, frames):
    """frames (B,k+1,128) -> per-token NLL (B,128) of the last frame given the earlier ones."""
    seq = to_sequence(frames)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(seq[:, :-1])
    lg = logits[:, -TOK_PER_FRAME:].float()                        # predictions for the last 128 tokens
    tgt = seq[:, -TOK_PER_FRAME:]
    assert (tgt != BOS).all()
    return F.cross_entropy(lg.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none").view(tgt.shape)


def prev_token_table(tokens, nseg, k=0.05):
    """Same add-k prev-token model as baselines.py, built with bincount (fast)."""
    C = np.zeros(VOCAB * VOCAB, np.int64)
    for s in range(nseg):
        t = np.asarray(tokens[s], np.int64)
        C += np.bincount((t[:-1] * VOCAB + t[1:]).ravel(), minlength=VOCAB * VOCAB)
    C = C.reshape(VOCAB, VOCAB) + k
    return np.log(C / C.sum(1, keepdims=True))


def boot_ci(x, n=2000, seed=0):
    """95% bootstrap CI of the mean of per-window values x."""
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base")
    ap.add_argument("--token-dir", default="data/tokens")
    ap.add_argument("--train", nargs="+", default=["data-0000", "data-0001", "data-0002"])
    ap.add_argument("--test", default="data-0039")
    ap.add_argument("--eval-seg", type=int, default=400)
    ap.add_argument("--windows", type=int, default=2048)
    ap.add_argument("--contexts", type=int, nargs="+", default=[1, 2, 4, 8, 15])
    ap.add_argument("--fit-seg", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=4321)
    a = ap.parse_args()

    out = Path("runs") / a.run
    ck = torch.load(out / "best.pt", map_location="cuda", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = WorldModel(cfg).cuda().eval()
    model.load_state_dict(ck["model"])
    assert max(a.contexts) <= cfg.frames - 1
    td = Path(a.token_dir)
    test = Windows(load_tokens(td, [a.test])[: a.eval_seg], cfg.frames)
    pairs = test.fixed(a.windows, seed=a.seed)
    win = np.stack([test.t[s, st:st + cfg.frames] for s, st in pairs]).astype(np.int64)  # (W,16,128)
    tgt, prev = win[:, -1], win[:, -2]
    rep = tgt == prev                                               # (W,128) repeat mask

    L = prev_token_table(load_tokens(td, a.train), a.fit_seg)
    base = -L[prev, tgt]                                            # (W,128)

    def summary(nll):
        per_win = nll.mean(1)
        return {"all": float(nll.mean()), "all_ci95": boot_ci(per_win),
                "repeat": float(nll[rep].mean()), "changed": float(nll[~rep].mean())}

    res = {"step": ck["step"], "windows": a.windows, "seed": a.seed, "repeat_rate": float(rep.mean()),
           "prev-token": summary(base), "context": {}}
    per_k = {}
    for k in a.contexts:
        chunks = []
        for i in range(0, len(win), a.batch):
            x = torch.from_numpy(win[i:i + a.batch, -(k + 1):]).cuda()
            chunks.append(target_nll(model, x).cpu().numpy())
        per_k[k] = np.concatenate(chunks)
        res["context"][k] = summary(per_k[k])
        r = res["context"][k]
        print(f"k={k:2d}  all {r['all']:.4f} [{r['all_ci95'][0]:.4f}, {r['all_ci95'][1]:.4f}]  "
              f"repeat {r['repeat']:.4f}  changed {r['changed']:.4f}")
    k0 = a.contexts[0]
    res["gain_vs_k%d" % k0] = {k: {"all": float((per_k[k0] - per_k[k]).mean()),
                                   "all_ci95": boot_ci((per_k[k0] - per_k[k]).mean(1)),
                                   "repeat": float((per_k[k0] - per_k[k])[rep].mean()),
                                   "changed": float((per_k[k0] - per_k[k])[~rep].mean())}
                               for k in a.contexts[1:]}
    b = res["prev-token"]
    print(f"prev-token  all {b['all']:.4f}  repeat {b['repeat']:.4f}  changed {b['changed']:.4f}  "
          f"(repeat rate {res['repeat_rate']:.3f})")
    (out / "ablation.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out / 'ablation.json'}")


if __name__ == "__main__":
    main()
