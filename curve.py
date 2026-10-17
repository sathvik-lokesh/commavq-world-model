"""Learning curves of runs/base and runs/pose (held-out NLL per eval) against the prev-token baseline.

Writes runs/curve.png.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run):
    """runs/<run>/log.jsonl -> (steps, test NLL), skipping step 0 (7.05, flattens the plot)."""
    rows = [json.loads(l) for l in open(f"runs/{run}/log.jsonl")]
    rows = [r for r in rows if r["step"] > 0]
    return [r["step"] for r in rows], [r["test_nll"] for r in rows]


def main():
    base = json.load(open("runs/baselines.json"))["prev-token"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for run, label in [("base", "base (video only)"), ("pose", "pose-conditioned")]:
        ax.plot(*load(run), marker=".", label=label)
    ax.axhline(base, ls="--", c="gray", label=f"prev-token baseline ({base:.2f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out NLL (nats/token)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("runs/curve.png", dpi=120)
    print("wrote runs/curve.png")


if __name__ == "__main__":
    main()
