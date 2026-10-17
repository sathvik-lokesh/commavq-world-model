"""M4: does the pose-conditioned model actually use the pose? Run: make poseeval

Same 2048 windows and target frame as ablate.py (frame 15, full 15-frame context). Scores
  base       the unconditioned model (runs/base)
  pose-real  the pose model with the true ego motion
  pose-shuf  the pose model with ego motion taken from another window (fixed permutation)
on the same target tokens, split into repeat/changed. If pose-shuf is no worse than pose-real,
the model ignores the pose. Caveat: comma estimated the poses from the video, so the true pose
carries some information about the target frame itself; a lower loss is expected and is not on
its own proof of a world model (the steering dreams test that).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ablate import boot_ci
from src.data import Windows, load_pose, load_tokens
from src.model import TOK_PER_FRAME, VOCAB, Config, WorldModel, to_sequence


def load(run):
    ck = torch.load(Path("runs") / run / "best.pt", map_location="cuda", weights_only=False)
    m = WorldModel(Config(**ck["cfg"])).cuda().eval()
    m.load_state_dict(ck["model"])
    return m, ck["step"]


@torch.no_grad()
def nll(model, win, act, batch):
    """win (W,16,128), act (W,15,6) or None -> per-token NLL (W,128) of the last frame."""
    out = []
    for i in range(0, len(win), batch):
        x = to_sequence(torch.from_numpy(win[i:i + batch]).cuda())
        a = None if act is None else torch.from_numpy(act[i:i + batch]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x[:, :-1], a) if a is not None else model(x[:, :-1])
        tgt = x[:, -TOK_PER_FRAME:]
        out.append(F.cross_entropy(lg[:, -TOK_PER_FRAME:].float().reshape(-1, VOCAB), tgt.reshape(-1),
                                   reduction="none").view(tgt.shape).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base")
    ap.add_argument("--run", default="pose")
    ap.add_argument("--token-dir", default="data/tokens")
    ap.add_argument("--test", default="data-0039")
    ap.add_argument("--eval-seg", type=int, default=400)
    ap.add_argument("--windows", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    td = Path(a.token_dir)
    pm, pstep = load(a.run)
    bm, bstep = load(a.base)
    W = pm.c.frames
    test = Windows(load_tokens(td, [a.test])[: a.eval_seg], W, pose=load_pose(td, [a.test])[: a.eval_seg])
    pairs = test.fixed(a.windows, seed=a.seed)                         # same windows as ablate.py
    win, act = (t.cpu().numpy() for t in test.take(pairs, device="cpu"))
    act = act.astype(np.float32)
    shuf = act[np.random.default_rng(0).permutation(len(act))]
    rep = win[:, -1] == win[:, -2]

    res = {"run_step": pstep, "base_step": bstep, "windows": a.windows, "seed": a.seed}
    scores = {"base": nll(bm, win, None, a.batch), "pose-real": nll(pm, win, act, a.batch),
              "pose-shuf": nll(pm, win, shuf, a.batch)}
    for k, v in scores.items():
        res[k] = {"all": float(v.mean()), "all_ci95": boot_ci(v.mean(1)),
                  "repeat": float(v[rep].mean()), "changed": float(v[~rep].mean())}
        print(f"{k:10s} all {v.mean():.4f}  repeat {v[rep].mean():.4f}  changed {v[~rep].mean():.4f}")
    for x, y in [("base", "pose-real"), ("pose-shuf", "pose-real")]:
        d = (scores[x] - scores[y]).mean(1)
        res[f"gain_{y}_over_{x}"] = {"mean": float(d.mean()), "ci95": boot_ci(d)}
        print(f"{y} better than {x} by {d.mean():.4f} nats/token, 95% CI {boot_ci(d)}")
    (Path("runs") / a.run / "pose_eval.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
