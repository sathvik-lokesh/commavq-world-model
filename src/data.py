"""commaVQ shards -> flat uint16 memmaps, and a random-window sampler.

A shard (data-XXXX.tar.gz) holds ~2400 one-minute segments, each as
  <id>.token.npy (1200, 8, 16) int16  in [0, 1023]   - 20 fps VQ frames
  <id>.pose.npy  (1200, 6)  float32                  - per-frame pose (kept, not used yet)
  <id>.json      metadata
`prepare_shard` writes data/tokens/data-XXXX.{tokens,pose}.bin + .json index, so training
reads straight from disk with no decompression.
"""
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

FRAMES_PER_SEG = 1200


def prepare_shard(tar_path: Path, out_dir: Path) -> dict:
    """Stream one tar.gz into memmaps. Returns the index dict (also written to <stem>.json)."""
    stem = tar_path.name.removesuffix(".tar.gz")
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_f = open(out_dir / f"{stem}.tokens.bin", "wb")
    pose_f = open(out_dir / f"{stem}.pose.bin", "wb")
    ids, pending = [], {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith(".npy"):
                continue
            seg, kind = m.name.rsplit(".", 2)[0], m.name.rsplit(".", 2)[1]
            arr = np.load(io.BytesIO(tar.extractfile(m).read()))  # numpy 2.5 needs a real file obj
            pending.setdefault(seg, {})[kind] = arr
            if len(pending[seg]) == 2:                              # both token + pose seen
                d = pending.pop(seg)
                t, p = d["token"], d["pose"]
                assert t.shape == (FRAMES_PER_SEG, 8, 16) and 0 <= t.min() and t.max() < 1024, (seg, t.shape)
                assert p.shape == (FRAMES_PER_SEG, 6), (seg, p.shape)
                tok_f.write(t.astype(np.uint16).reshape(FRAMES_PER_SEG, 128).tobytes())
                pose_f.write(p.astype(np.float32).tobytes())
                ids.append(seg)
    tok_f.close(); pose_f.close()
    assert not pending, f"{len(pending)} segments missing token or pose"
    index = {"shard": stem, "segments": ids, "n": len(ids)}
    (out_dir / f"{stem}.json").write_text(json.dumps(index))
    return index


def load_tokens(token_dir: Path, shards) -> np.ndarray:
    """Concatenate shards as one (N_seg, 1200, 128) uint16 view (memmapped, ~0.3 MB per segment)."""
    arrs = []
    for s in shards:
        n = json.loads((token_dir / f"{s}.json").read_text())["n"]
        arrs.append(np.memmap(token_dir / f"{s}.tokens.bin", np.uint16, "r", shape=(n, FRAMES_PER_SEG, 128)))
    return arrs[0] if len(arrs) == 1 else np.concatenate(arrs)


def load_pose(token_dir: Path, shards) -> np.ndarray:
    """(N_seg, 1200, 6) float32 per-frame ego motion, same segment order as load_tokens.
    Verified from the data (undocumented upstream): [vx, vy, vz] m/s, [roll, pitch, yaw] rate rad/s in
    openpilot's car frame (x forward, y right, z down, so yaw > 0 = turning right). pose[t] is the
    motion from frame t to t+1; pose[1199] is always NaN."""
    arrs = []
    for s in shards:
        n = json.loads((token_dir / f"{s}.json").read_text())["n"]
        arrs.append(np.memmap(token_dir / f"{s}.pose.bin", np.float32, "r", shape=(n, FRAMES_PER_SEG, 6)))
    return arrs[0] if len(arrs) == 1 else np.concatenate(arrs)


class Windows:
    """Random (segment, start) windows of `frames` consecutive frames, every `stride`-th frame.
    With `pose`, batch/take return (tokens, actions): actions (B, frames-1, 6) = pose[start+i-1],
    the motion that leads into frame i, for i = 1..frames-1."""

    def __init__(self, tokens: np.ndarray, frames: int, stride: int = 1, seed: int = 0, pose=None):
        assert pose is None or stride == 1, "pose conditioning assumes consecutive frames"
        self.t, self.p, self.frames, self.stride = tokens, pose, frames, stride
        self.span = (frames - 1) * stride + 1
        self.rng = np.random.default_rng(seed)

    def batch(self, B: int, device="cuda") -> torch.Tensor:
        """(B, frames, 128) int64 on device."""
        seg = self.rng.integers(0, len(self.t), B)
        start = self.rng.integers(0, FRAMES_PER_SEG - self.span + 1, B)
        return self.take(list(zip(seg, start)), device)

    def fixed(self, n: int, seed: int = 1234):
        """A deterministic list of (seg, start) pairs: the same eval windows on every run."""
        r = np.random.default_rng(seed)
        return list(zip(r.integers(0, len(self.t), n), r.integers(0, FRAMES_PER_SEG - self.span + 1, n)))

    def take(self, pairs, device="cuda") -> torch.Tensor:
        """Materialise fixed (seg, start) pairs as a (len, frames, 128) batch."""
        x = np.stack([self.t[s, a:a + self.span:self.stride] for s, a in pairs])
        x = torch.from_numpy(x.astype(np.int64)).to(device, non_blocking=True)
        if self.p is None:
            return x
        act = np.stack([self.p[s, a:a + self.frames - 1] for s, a in pairs])
        assert not np.isnan(act).any()
        return x, torch.from_numpy(act).to(device, non_blocking=True)
