"""SFT warmup / cleanup reactivation comparison plot for the paper.

Two square panels in a horizontal layout:
  (a) EM GPT-OSS-20B: misalignment rate vs. Type-1 reactivation N
  (b) MMLU/BCOT GPT-OSS-20B: exploit rate vs. Type-1 reactivation N

Run:
    python plots/sft_warmup_reactivation_comparison.py

Produces:
    docs/result_collection/0506/paper/images/sft_warmup_reactivation_comparison.pdf
    plots/sft_warmup_reactivation_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

x = [0, 1, 2]
x_labels = ["N=0", "N=500", "N=2k"]

# EM setup: misalignment rate (%). Parenthesized alignment scores from the table
# are intentionally not plotted; the panel focuses on reactivation.
em_misalign = {
    "Base": [1.50, 2.50, 16.25],
    "SFT (full)": [3.62, 9.62, 7.38],
    "SFT warmup only": [4.50, 9.75, 8.50],
    "SFT + PEARL": [0.00, 0.00, 0.10],
}

# MMLU/BCOT setup: exploit rate (%). Table entries are
# clean acc. / cued acc. / exploit rate; only exploit is plotted here.
mmlu_exploit = {
    "Base": [26.10, 29.60, 26.40],
    "SFT (full)": [6.00, 11.50, 35.70],
    "SFT warmup only": [4.00, 8.40, 13.50],
    "SFT + PEARL": [3.60, 7.40, 16.40],
}


# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

STYLE = {
    "Base": dict(color="#777777", ls=":", marker="D"),
    "SFT (full)": dict(color="#C00000", ls="--", marker="s"),
    "SFT warmup only": dict(color="#FF6B6B", ls="--", marker="s"),
    "SFT + PEARL": dict(color="#2F3192", ls="-", marker="o"),
}

LW = 2.4
MS = 7
FONT = 11
TITLE_FONT = 12
LABEL_FONT = 11


def _plot_panel(ax, data, ylabel: str, title: str, ymax: float) -> None:
    ax.set_facecolor("white")
    for name, vals in data.items():
        s = STYLE[name]
        ax.plot(
            x,
            vals,
            color=s["color"],
            linestyle=s["ls"],
            linewidth=LW,
            marker=s["marker"],
            markersize=MS,
            zorder=3,
            clip_on=False,
            label=name,
        )

    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")

    ax.set_xlim(-0.25, max(x) + 0.25)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=FONT)
    ax.tick_params(axis="y", labelsize=FONT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT)
    ax.set_xlabel("Reactivation SFT examples (N)", fontsize=LABEL_FONT)
    ax.set_title(title, fontsize=TITLE_FONT, fontweight="bold", pad=8)
    ax.set_box_aspect(1)


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45))
fig.patch.set_facecolor("white")

_plot_panel(
    axes[0],
    em_misalign,
    "Misalignment rate",
    "(a) Broad-trigger organism",
    ymax=18,
)
_plot_panel(
    axes[1],
    mmlu_exploit,
    "Exploit rate",
    "(b) Narrow-trigger organism",
    ymax=40,
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    fontsize=FONT - 1,
    frameon=False,
    bbox_to_anchor=(0.5, -0.005),
    handlelength=2.0,
    columnspacing=1.2,
)

plt.tight_layout(rect=[0, 0.10, 1, 1])


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

THIS_FILE = Path(__file__).resolve()
PROJECT = THIS_FILE.parents[1]
PNG = THIS_FILE.parent / "sft_warmup_reactivation_comparison.png"
PDF = (
    PROJECT
    / "docs"
    / "result_collection"
    / "0506"
    / "paper"
    / "images"
    / "sft_warmup_reactivation_comparison.pdf"
)
PDF.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(PDF, bbox_inches="tight")
fig.savefig(PNG, dpi=300, bbox_inches="tight")
print(f"Saved {PDF}")
print(f"Saved {PNG}")
