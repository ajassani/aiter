"""Figures for the explainer. Launch shapes come from segments.py; timings are the measured microbenchmark numbers.

    python make_figures.py   # writes img/*.png next to this file
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from segments import CUS, DEFAULT, NUM_WARPS, PR, TILE, launch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)
GREY, BLUE, ORANGE = "#bbbbbb", "#1f77b4", "#ff7f0e"
CTX = 131072

# measured, MI325X, batch 1, CUDA graphs, through aiter's config lookup (us per call)
MEASURED = [("verify\n131k", 128, 92, 131072), ("verify\n196k", 184, 115, 196608),
            ("draft\n131k", 120, 69, 131072), ("draft\n196k", 181, 83, 196608)]


def fig_occupancy():
    """Programs per CU for a batch-1 draft call at 131k, assuming even round-robin dispatch."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6))
    cmap = {0: "#eeeeee", 1: "#9ecae1", 2: "#2171b5", 3: "#08306b"}
    for ax, (name, table) in zip(axes, (("default config", DEFAULT), ("this PR", PR))):
        r = launch(table["draft"], 1, 1, CTX)
        programs = r["live"]
        per_cu = [programs // CUS + (1 if i < programs % CUS else 0) for i in range(CUS)]
        for i, n in enumerate(per_cu):
            xcd, cu = divmod(i, 38)
            ax.add_patch(Rectangle((cu, 7 - xcd), 0.9, 0.9, color=cmap[min(n, 3)]))
        ax.set_xlim(0, 38)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")
        idle = per_cu.count(0)
        ax.set_title(f"{name}: {r['seg']} segments = {programs} programs ({programs * NUM_WARPS} waves) on 304 CUs; "
                     f"{idle} CUs idle", fontsize=10, loc="left")
    fig.legend(handles=[Patch(color=cmap[k], label=l) for k, l in
                        ((0, "idle CU"), (1, "1 program (2 waves)"), (2, "2 programs (4 waves)"))],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Batch-1 draft attention at 131k context, one row per XCD (38 CUs each)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(os.path.join(OUT, "occupancy.png"), dpi=150)


def fig_segments_vs_batch():
    batches = [1, 2, 4, 8, 16, 32]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for table, color, name in ((DEFAULT, GREY, "default"), (PR, BLUE, "this PR")):
        for call, qlen, style in (("verify", 4, "-o"), ("draft", 1, "--s")):
            rs = [launch(table[call], b, qlen, CTX) for b in batches]
            a1.plot(batches, [r["live"] * NUM_WARPS for r in rs], style, color=color, label=f"{name}, {call}")
            a2.plot(batches, [r["reduce_mb"] for r in rs], style, color=color, label=f"{name}, {call}")
    a1.axhline(CUS * 4, color="k", lw=0.8, ls=":")
    a1.text(1.05, CUS * 4 * 1.08, "1 wave per SIMD (1,216)", fontsize=8)
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches, [str(b) for b in batches])
        ax.set_xlabel("batch size (sequences)")
        ax.grid(alpha=0.3)
    a1.set_yscale("log", base=2)
    a1.set_ylabel("live waves launched")
    a1.set_title("Parallelism: the PR holds about 2,048 waves at every batch", fontsize=10)
    a2.set_ylabel("fp32 partials written for the reduce (MB)")
    a2.set_title("Reduce traffic: the default grows with batch", fontsize=10)
    a1.legend(fontsize=8)
    fig.suptitle(f"Split-KV launch at {CTX:,}-token context (Qwen3.5 TP4 shapes, 304 CUs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "segments_vs_batch.png"), dpi=150)


def fig_measured():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    x = range(len(MEASURED))
    w = 0.38
    a1.bar([i - w / 2 for i in x], [m[1] for m in MEASURED], w, color=GREY, label="default")
    a1.bar([i + w / 2 for i in x], [m[2] for m in MEASURED], w, color=BLUE, label="this PR")
    for i, m in enumerate(MEASURED):
        a1.text(i + w / 2, m[2] + 3, f"-{100 * (1 - m[2] / m[1]):.0f}%", ha="center", fontsize=9)
    a1.set_xticks(list(x), [m[0] for m in MEASURED])
    a1.set_ylabel("microseconds per call (batch 1)")
    a1.set_title("Measured attention call time", fontsize=10)
    a1.legend(fontsize=8)
    for name, table, color, idx in (("default", DEFAULT, GREY, 1), ("this PR", PR, BLUE, 2)):
        tiles = [launch(table["verify" if m[0].startswith("verify") else "draft"], 1,
                        4 if m[0].startswith("verify") else 1, m[3])["tiles"] for m in MEASURED]
        a2.scatter(tiles, [m[idx] for m in MEASURED], color=color, s=50, label=name, zorder=3)
    xs = [0, 100]
    a2.plot(xs, [1.9 * v for v in xs], color="k", lw=0.8, ls=":")
    a2.text(55, 1.9 * 55 + 12, "1.9 us per tile", fontsize=8, rotation=33)
    a2.set_xlabel("tiles each segment walks serially (16 tokens per tile)")
    a2.set_ylabel("microseconds per call")
    a2.set_title("Default sits on the serial-chain line; the PR leaves it", fontsize=10)
    a2.set_xlim(0, 105)
    a2.set_ylim(0, 200)
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    fig.suptitle("MI325X, Qwen3.5 TP4 shapes, bf16 KV, page 16, CUDA graphs", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "measured.png"), dpi=150)


def fig_split_schematic():
    """Gantt view of the first 8,192 tokens: which program covers which tile, and at which serial step."""
    span = 8192
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, (name, table, color) in zip(axes, (("default", DEFAULT, GREY), ("this PR", PR, BLUE))):
        r = launch(table["draft"], 1, 1, CTX)
        rows = span // (r["tiles"] * TILE)
        for p in range(rows):
            for t in range(r["tiles"]):
                ax.add_patch(Rectangle((t, rows - 1 - p), 0.85, 0.8, color=color))
        ax.set_xlim(0, 66)
        ax.set_ylim(-0.5, 33)
        ax.set_yticks([])
        ax.set_xlabel("serial step inside a program (one 16-token tile per step)")
        ax.set_title(f"{name}: {rows} of {r['seg']} programs cover these tokens,\n"
                     f"each walks {r['tiles']} tiles in a row", fontsize=10)
    axes[0].text(33, 12, "programs run in parallel (rows);\ntiles inside a program run one after another (columns)",
                 ha="center", fontsize=8)
    fig.suptitle("The first 8,192 tokens of a 131k context, draft call at batch 1: same tiles, "
                 "4x the programs, 1/4 the chain", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "split_schematic.png"), dpi=150)


if __name__ == "__main__":
    fig_occupancy()
    fig_segments_vs_batch()
    fig_measured()
    fig_split_schematic()
    print("wrote", sorted(os.listdir(OUT)))
