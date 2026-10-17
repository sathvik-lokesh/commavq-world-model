"""Train the world model. Run: make train  (or python -E -s train.py --help)

Logs to runs/<name>/log.jsonl, keeps runs/<name>/best.pt (by held-out loss).
Held-out = fixed windows from the test shard's first --eval-seg segments, the same segments
baselines.py scores, so the numbers sit on one scale.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.data import Windows, load_pose, load_tokens
from src.model import Config, WorldModel, frame_loss


def as_args(b):
    """Windows returns tokens, or (tokens, actions) with pose: always hand frame_loss a tuple."""
    return b if isinstance(b, tuple) else (b,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--token-dir", default="data/tokens")
    ap.add_argument("--train", nargs="+", default=["data-0000", "data-0001", "data-0002"])
    ap.add_argument("--test", default="data-0039")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1, help="use every k-th frame (k=2 -> 10 fps)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--eval-seg", type=int, default=400)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    ap.add_argument("--pose", action="store_true", help="M4: condition each frame on the ego motion into it")
    a = ap.parse_args()

    out = Path("runs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    if a.resume and (out / "args.json").exists():                     # resume = same run: restore its flags (e.g. --pose),
        given = {t.split("=")[0][2:].replace("-", "_") for t in sys.argv[1:] if t.startswith("--")}  # flags typed now win
        vars(a).update({k: v for k, v in json.loads((out / "args.json").read_text()).items() if k not in given})
    torch.manual_seed(a.seed)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    td = Path(a.token_dir)
    tr_pose = load_pose(td, a.train) if a.pose else None
    te_pose = load_pose(td, [a.test])[: a.eval_seg] if a.pose else None
    train = Windows(load_tokens(td, a.train), a.frames, a.stride, a.seed, pose=tr_pose)
    test = Windows(load_tokens(td, [a.test])[: a.eval_seg], a.frames, a.stride, pose=te_pose)
    eval_pairs = test.fixed(a.eval_windows)

    act = {}
    if a.pose:                                                        # normalisation fit on train only
        sample = np.asarray(tr_pose[:1500, :-1]).reshape(-1, 6)
        act = dict(action_dim=6, act_mean=tuple(map(float, sample.mean(0))), act_std=tuple(map(float, sample.std(0))))
    cfg = Config(n_layer=a.layers, n_head=a.heads, d_model=a.d_model, frames=a.frames, **act)
    model = WorldModel(cfg).cuda()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, context {cfg.block} tokens, "
          f"{len(train.t)} train segments")
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    fwd = torch.compile(model) if a.compile else model

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup))))

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = 0.0
        for i in range(0, len(eval_pairs), a.batch):
            b = as_args(test.take(eval_pairs[i:i + a.batch]))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot += frame_loss(fwd, *b).item() * len(b[0])
        model.train()
        return tot / len(eval_pairs)

    best, start = float("inf"), 0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck["best"]                            # redo that step's update; its eval is skipped below
        train.rng = np.random.default_rng([a.seed, start])            # fresh, reproducible stream after resume
        print(f"resumed at step {start}, best so far {best:.4f}")
    log, t0 = open(out / "log.jsonl", "a"), time.time()
    for step in range(start, a.steps + 1):
        if step % a.eval_every == 0 and not (a.resume and step == start and step > 0):
            ev = evaluate()
            rec = {"step": step, "test_nll": ev, "elapsed_min": (time.time() - t0) / 60,
                   "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            if ev < best:
                best = ev
                torch.save({"cfg": vars(cfg), "model": model.state_dict(), "step": step, "test_nll": ev, "args": vars(a)},
                           out / "best.pt")
            # resumable state: stop any time, lose at most --eval-every steps
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best},
                       out / "last.tmp"); (out / "last.tmp").replace(out / "last.pt")
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec))
        if step == a.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(a.accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = frame_loss(fwd, *as_args(train.batch(a.batch))) / a.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            print(f"step {step} loss {loss.item() * a.accum:.4f} lr {lr_at(step):.2e} {(time.time() - t0) / 60:.1f} min")
    print(f"best test nll {best:.4f}")


if __name__ == "__main__":
    main()
