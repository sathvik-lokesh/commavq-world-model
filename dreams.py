"""M3: how good are the dreams, measured over many clips instead of picked by eye. Run: make dreams

Clips are drawn by a fixed seed from held-out segments 400+ (never trained on, never used for
the NLL scores) and written to dreams/clips.json BEFORE anything is generated. Each clip gets
--context real frames, then the model imagines --future frames at temperature 1, --seeds times.

Per horizon step we compare, in pixels after decoding, the dream against the real future and
against a "frozen" baseline that just repeats the last real frame. Beating frozen means the
model predicts how the scene moves, not only what it looks like. We also track the motion rate
(share of tokens that change from one frame to the next): a dream that freezes goes to 0, one
that dissolves into noise goes towards 1; real driving sits near 0.6.

Videos (top = real, bottom = dream; green bar = real context, red = imagined): clips 0-3 with
seed 0 (fixed in advance), plus the best and the worst clip by dream/frozen error ratio.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import matplotlib
import numpy as np
import onnxruntime as ort
import torch

from render import decode
from src.data import FRAMES_PER_SEG, load_pose, load_tokens
from src.model import Config, WorldModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def motion(x):
    """(..., F, 128) tokens -> (..., F-1) share of tokens that differ from the previous frame."""
    return (x[..., 1:, :] != x[..., :-1, :]).mean(-1)


def hshift(a, b):
    """Horizontal shift (px) of frame b vs a by phase correlation on the upper half (sky/horizon).
    Positive = scene moved left = camera turned right (checked on a synthetic roll, see EXPERIMENTS.md)."""
    a, b = a[:64].mean(-1), b[:64].mean(-1)
    A, B = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = np.fft.ifft2(A * np.conj(B) / (np.abs(A * np.conj(B)) + 1e-9)).real
    dx = np.unravel_index(r.argmax(), r.shape)[1]
    return dx if dx < a.shape[1] // 2 else dx - a.shape[1]


def turn(px):
    """(F,H,W,3) frames -> summed frame-to-frame horizontal shift (px); > 0 = turning right."""
    return float(sum(hshift(px[k], px[k + 1]) for k in range(len(px) - 1)))


def video(path, real, dream, context, fps=20):
    """Stack real (top) over dream (bottom), mark context/imagined frames, write mp4."""
    dream = dream.copy()
    dream[:context, :3] = [0, 200, 0]
    dream[context:, :3] = [220, 40, 40]
    iio.mimwrite(path, list(np.concatenate([real, dream], 1)), fps=fps, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base")
    ap.add_argument("--token-dir", default="data/tokens")
    ap.add_argument("--test", default="data-0039")
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--context", type=int, default=15)
    ap.add_argument("--future", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min-seg", type=int, default=400, help="segments below this were used for scoring")
    ap.add_argument("--clip-seed", type=int, default=2026)
    ap.add_argument("--out", default="dreams", help="subfolder of runs/<run>/")
    ap.add_argument("--steer", default="real", choices=["real", "left", "right", "stop"],
                    help="pose models only: actions for the imagined frames (context always gets the real ones)")
    ap.add_argument("--yaw", type=float, default=0.2, help="rad/s for --steer left/right (p99 of real |yaw| ~0.22)")
    ap.add_argument("--videos", type=int, default=1, help="0 = metrics only")
    a = ap.parse_args()

    out = Path("runs") / a.run / a.out
    out.mkdir(parents=True, exist_ok=True)
    T = load_tokens(Path(a.token_dir), [a.test])
    span = a.context + a.future
    r = np.random.default_rng(a.clip_seed)
    clips = [(int(s), int(st)) for s, st in zip(r.integers(a.min_seg, len(T), a.clips),
                                                 r.integers(0, FRAMES_PER_SEG - span + 1, a.clips))]
    (out / "clips.json").write_text(json.dumps({"clips": clips, **vars(a)}, indent=2))   # fixed before generating

    ck = torch.load(Path("runs") / a.run / "best.pt", map_location="cuda", weights_only=False)
    model = WorldModel(Config(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    W = model.c.frames
    assert a.context <= W - 1

    real = np.stack([np.asarray(T[s, st:st + span], np.int64) for s, st in clips])        # (C,span,128)
    C, S = len(clips), a.seeds
    seq = torch.from_numpy(real[:, :a.context]).cuda().repeat_interleave(S, 0)          # (C*S,ctx,128), clip-major
    act = None
    if model.c.action_dim:                       # act[:, i] = motion into frame i+1 (load_pose docstring)
        P = load_pose(Path(a.token_dir), [a.test])
        act = np.stack([np.asarray(P[s, st:st + span - 1], np.float32) for s, st in clips])   # (C,span-1,6)
        imag = slice(a.context - 1, None)                                                  # into imagined frames
        if a.steer in ("left", "right"):
            act[:, imag, 5] = a.yaw if a.steer == "right" else -a.yaw                      # yaw > 0 = right
        elif a.steer == "stop":
            act[:, imag] = 0.0
        act = torch.from_numpy(act).cuda().repeat_interleave(S, 0)
    else:
        assert a.steer == "real", "--steer needs a pose-conditioned run"
    torch.manual_seed(0)
    for i in range(a.future):
        j = seq.shape[1]                                                                  # index of the new frame
        ctx_act = None if act is None else act[:, max(0, j - W + 1):j]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt = model.generate_frame(seq[:, -(W - 1):], a.temperature, act=ctx_act)
        seq = torch.cat([seq, nxt[:, None]], 1)
        if i % 10 == 9:
            print(f"generated {i + 1}/{a.future} frames", flush=True)
    dream = seq.cpu().numpy().reshape(C, S, span, 128)

    sess = ort.InferenceSession("data/models/decoder.onnx", providers=["CPUExecutionProvider"])
    ctx_last = a.context - 1
    real_px = np.stack([decode(sess, real[c, ctx_last:]) for c in range(C)]).astype(np.float32)  # (C,1+fut,H,W,3)
    frozen_err = np.abs(real_px[:, 1:] - real_px[:, :1]).mean((2, 3, 4))                       # (C,fut)
    dream_err = np.zeros((C, S, a.future), np.float32)
    dream_turn, real_turn = np.zeros((C, S)), np.array([turn(real_px[c]) for c in range(C)])
    for c in range(C):
        for s in range(S):
            px = decode(sess, dream[c, s, a.context:]).astype(np.float32)
            dream_err[c, s] = np.abs(px - real_px[c, 1:]).mean((1, 2, 3))
            dream_turn[c, s] = turn(np.concatenate([real_px[c, :1], px]))
        print(f"decoded clip {c + 1}/{C}", flush=True)

    ratio = dream_err.mean((1, 2)) / frozen_err.mean(1)                                    # per clip, <1 = beats frozen
    real_motion = motion(real[:, ctx_last:])                                               # (C,fut)
    dream_motion = motion(dream[:, :, ctx_last:])                                          # (C,S,fut)
    beats = (dream_err.mean(1) < frozen_err)                                               # (C,fut) seed-mean vs frozen
    res = {
        "step": ck["step"], "clips": clips, "context": a.context, "future": a.future, "seeds": S,
        "temperature": a.temperature, "steer": a.steer, "yaw": a.yaw,
        "turn_px": {"dream_mean": float(dream_turn.mean()), "real_mean": float(real_turn.mean()),
                    "dream_per_clip": dream_turn.mean(1).tolist(), "real_per_clip": real_turn.tolist()},
        "pixel_mae_by_horizon": {"dream": dream_err.mean((0, 1)).tolist(), "frozen": frozen_err.mean(0).tolist()},
        "motion_by_horizon": {"dream": dream_motion.mean((0, 1)).tolist(), "real": real_motion.mean(0).tolist()},
        "share_clips_beating_frozen_by_horizon": beats.mean(0).tolist(),
        "per_clip": [{"seg": s, "start": st, "ratio": float(rt), "real_motion": float(real_motion[c].mean()),
                      "dream_motion": float(dream_motion[c].mean())}
                     for c, ((s, st), rt) in enumerate(zip(clips, ratio))],
        "clips_beating_frozen_overall": int((ratio < 1).sum()),
    }
    (out / "dreams.json").write_text(json.dumps(res, indent=2))

    t = (np.arange(a.future) + 1) / 20
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, res["pixel_mae_by_horizon"]["dream"], label="model dream")
    ax[0].plot(t, res["pixel_mae_by_horizon"]["frozen"], label="frozen last frame")
    ax[0].set(xlabel="seconds into the future", ylabel="pixel MAE vs real (0-255)", title="Error vs real future")
    ax[1].plot(t, res["motion_by_horizon"]["dream"], label="model dream")
    ax[1].plot(t, res["motion_by_horizon"]["real"], label="real")
    ax[1].set(xlabel="seconds into the future", ylabel="share of tokens changed", title="Motion rate", ylim=(0, 1))
    for x in ax:
        x.legend(); x.grid(alpha=0.3)
    fig.suptitle(f"{C} held-out clips x {S} seeds, {a.context} real frames of context")
    fig.tight_layout(); fig.savefig(out / "dreams.png", dpi=120)

    picks = {f"clip{c}": c for c in range(min(4, C))} if a.videos else {}
    if a.videos:
        picks["best"], picks["worst"] = int(ratio.argmin()), int(ratio.argmax())
    for name, c in picks.items():
        video(out / f"{name}_seg{clips[c][0]}_{clips[c][1]}.mp4", decode(sess, real[c]), decode(sess, dream[c, 0]), a.context)
    for p in res["per_clip"]:
        print(f"seg {p['seg']:4d} start {p['start']:4d}  dream/frozen {p['ratio']:.2f}  "
              f"motion real {p['real_motion']:.2f} dream {p['dream_motion']:.2f}")
    print(f"turn (summed px shift, >0 = right): dream {dream_turn.mean():+.1f}  real {real_turn.mean():+.1f}  [steer={a.steer}]")
    print(f"clips beating frozen: {res['clips_beating_frozen_overall']}/{C}; wrote {out}")


if __name__ == "__main__":
    main()
