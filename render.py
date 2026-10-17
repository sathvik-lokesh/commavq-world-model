"""Real vs imagined: give the model a few real frames, let it dream the next ones, decode both.

Run: .venv/bin/python -E -s render.py runs/<name>/best.pt --seg 5 --start 300 --future 40
Writes runs/<name>/dream_seg<seg>_<start>.mp4: top = real future, bottom = model rollout.
Uses comma's decoder.onnx (commaai/commavq-gpt2m, MIT) to turn tokens back into 256x128 frames.
"""
import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import onnxruntime as ort
import torch

from src.data import load_tokens
from src.model import Config, WorldModel


def decode(sess, tokens: np.ndarray) -> np.ndarray:
    """(F,128) tokens -> (F,128,256,3) uint8 frames."""
    out = []
    for i in range(0, len(tokens), 16):
        y = sess.run(None, {"encoding_indices": tokens[i:i + 16].reshape(-1, 8, 16).astype(np.int64)})[0]
        out.append(((y.transpose(0, 2, 3, 1) + 1) * 127.5).clip(0, 255).astype(np.uint8))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--token-dir", default="data/tokens")
    ap.add_argument("--test", default="data-0039")
    ap.add_argument("--seg", type=int, default=5)
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--context", type=int, default=8, help="real frames given to the model")
    ap.add_argument("--future", type=int, default=40, help="frames to imagine")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cuda")
    stride = ck["args"]["stride"]
    model = WorldModel(Config(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    W = model.c.frames
    torch.manual_seed(a.seed)

    T = load_tokens(Path(a.token_dir), [a.test])
    idx = a.start + stride * np.arange(a.context + a.future)
    real = torch.from_numpy(np.asarray(T[a.seg, idx], np.int64)).cuda()          # (ctx+fut,128)
    seq = real[: a.context].clone()
    for _ in range(a.future):                                                     # slide the window
        ctx = seq[-(W - 1):][None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt = model.generate_frame(ctx, a.temperature, a.top_k)
        seq = torch.cat([seq, nxt], 0)

    sess = ort.InferenceSession(str(Path(__file__).parent / "data/models/decoder.onnx"),
                                providers=["CPUExecutionProvider"])
    top, bottom = decode(sess, real.cpu().numpy()), decode(sess, seq.cpu().numpy())
    bottom[: a.context, :3] = [0, 200, 0]                                          # green bar = real context
    bottom[a.context:, :3] = [220, 40, 40]                                         # red bar = imagined
    frames = np.concatenate([top, bottom], 1)
    match = (seq[a.context:] == real[a.context:]).float().mean().item()
    out = Path(a.ckpt).parent / f"dream_seg{a.seg}_{a.start}.mp4"
    iio.mimwrite(out, list(frames), fps=20 / stride, macro_block_size=1)
    print(f"wrote {out}  ({a.context} real + {a.future} imagined frames; token match vs real future {match:.1%})")


if __name__ == "__main__":
    main()
