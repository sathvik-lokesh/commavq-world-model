"""Non-learned baselines for next-frame token prediction. Run: make baselines

All are scored the same way as the model: mean cross-entropy (nats/token) of frame t's tokens
given frames < t, on the held-out shard. Count tables are fit on the train shards only.

  uniform        log(1024) = 6.93: knows nothing
  unigram        P(token)                               - global token frequencies
  copy-mix       a*[token == same position last frame] + (1-a)*unigram, a fit on train
  prev-token     P(token | token at the same position in the previous frame), add-k smoothed
  prev-token+pos P(token | previous token, position)    - 128x1024x1024 counts (~0.5 GB as int32)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_tokens

V = 1024


def counts_prev(tokens, max_seg, per_pos=False):
    """Count (prev, cur) pairs at the same frame position over the first max_seg segments."""
    C = np.zeros((128, V, V) if per_pos else (V, V), np.int32)
    for s in range(min(max_seg, len(tokens))):
        t = np.asarray(tokens[s], np.int64)
        prev, cur = t[:-1], t[1:]                                  # (1199,128)
        if per_pos:
            np.add.at(C, (np.broadcast_to(np.arange(128), prev.shape), prev, cur), 1)
        else:
            np.add.at(C, (prev, cur), 1)
    return C


def score(tokens, nseg, logp_fn):
    """Mean NLL (nats/token) over all frame transitions in the first nseg test segments."""
    tot, n = 0.0, 0
    for s in range(min(nseg, len(tokens))):
        t = np.asarray(tokens[s], np.int64)
        lp = logp_fn(t[:-1], t[1:])                                 # (1199,128) log p of cur
        tot -= lp.sum(); n += lp.size
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="data/tokens")
    ap.add_argument("--train", nargs="+", default=["data-0000", "data-0001", "data-0002"])
    ap.add_argument("--test", default="data-0039")
    ap.add_argument("--fit-seg", type=int, default=1500, help="train segments used for count tables")
    ap.add_argument("--test-seg", type=int, default=400)
    ap.add_argument("--k", type=float, default=0.05, help="add-k smoothing")
    a = ap.parse_args()
    td = Path(a.token_dir)
    tr, te = load_tokens(td, a.train), load_tokens(td, [a.test])

    uni = np.bincount(np.asarray(tr[: a.fit_seg]).ravel(), minlength=V).astype(np.float64) + 1
    luni = np.log(uni / uni.sum())
    res = {"uniform": float(np.log(V)), "unigram": score(te, a.test_seg, lambda p, c: luni[c])}

    # copy-mix: pick a on a slice of train, then score test
    best = None
    for alpha in np.linspace(0.05, 0.95, 19):
        f = lambda p, c, al=alpha: np.log(al * (p == c) + (1 - al) * np.exp(luni[c]))
        nll = score(tr[a.fit_seg:], 100, f)
        best = min(best or (nll, alpha), (nll, alpha))
    al = best[1]
    res["copy-mix"] = score(te, a.test_seg, lambda p, c: np.log(al * (p == c) + (1 - al) * np.exp(luni[c])))
    res["copy-mix_alpha"] = float(al)

    C = counts_prev(tr, a.fit_seg).astype(np.float64) + a.k
    L = np.log(C / C.sum(1, keepdims=True))
    res["prev-token"] = score(te, a.test_seg, lambda p, c: L[p, c])
    del C, L

    C = counts_prev(tr, a.fit_seg, per_pos=True)
    pos = np.arange(128)
    R = C.sum(-1, dtype=np.int64)                                   # (128, V) row totals, computed once
    def lp_pos(p, c):
        num = C[pos, p, c].astype(np.float64) + a.k
        den = R[pos, p].astype(np.float64) + a.k * V
        return np.log(num / den)
    res["prev-token+pos"] = score(te, a.test_seg, lp_pos)
    res["repeat_rate_test"] = float(np.mean([(np.asarray(te[s])[1:] == np.asarray(te[s])[:-1]).mean() for s in range(a.test_seg)]))
    res.update(fit_segments=a.fit_seg, test_segments=a.test_seg, test_shard=a.test, smoothing_k=a.k)

    Path("runs").mkdir(exist_ok=True)
    Path("runs/baselines.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:18s} {v:.4f}" if isinstance(v, float) else f"{k:18s} {v}")


if __name__ == "__main__":
    main()
