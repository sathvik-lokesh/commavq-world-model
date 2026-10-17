"""Convert downloaded shards to memmaps. Run: make prepare

--delete-raw removes each tar.gz after a successful conversion (its sha256 was already checked
by download.sh and is re-fetchable). Saves ~0.5 GB per shard on disk.
"""
import argparse
import hashlib
import json
from pathlib import Path

from src.data import prepare_shard


def sha256(path: Path) -> str:
    """Streamed sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/tokens")
    ap.add_argument("--delete-raw", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    manifest_p = out / "manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    for tar in sorted(Path(a.raw_dir).glob("data-*.tar.gz")):
        stem = tar.name.removesuffix(".tar.gz")
        if (out / f"{stem}.json").exists():
            print(f"skip {stem} (already prepared)")
        else:
            src_hash = sha256(tar)
            idx = prepare_shard(tar, out)
            manifest[stem] = {"segments": idx["n"], "source_sha256": src_hash,
                              "tokens_sha256": sha256(out / f"{stem}.tokens.bin")}
            manifest_p.write_text(json.dumps(manifest, indent=2))
            print(f"{stem}: {idx['n']} segments")
        if a.delete_raw:
            tar.unlink()
            print(f"deleted {tar}")


if __name__ == "__main__":
    main()
