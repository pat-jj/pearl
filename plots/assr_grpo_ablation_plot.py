"""PEARL (formerly ASSR) group-size / cached-prefix ablation plot for the paper.

Two panels, side-by-side, both on GPT-OSS-20B (Tinker):
  (a) narrow-trigger organism   - exploit rate vs N (N=0..2k)
  (b) entangled organism        - misalignment rate vs N (N=0..18k)

Run:
    python plots/assr_grpo_ablation_plot.py

Produces:
    docs/result_collection/0506/paper/images/assr_grpo_ablation.pdf
    plots/assr_grpo_ablation.png  (preview)

Data sources (raw):
  Narrow / GPT-OSS-20B (Tinker):
    - results/bcot_type1/bcot_t1_gptoss_g4_p1_n*.json
    - results/bcot_type1/bcot_t1_gptoss_g8_p2_n*.json (n1k/n2k re-evaluated 2026-05-06)
    - results/bcot_type1/bcot_t1_assr_no_sft_n*.json   (= PEARL g4_p2 default, no warmup, lr 2e-5)
    - results/bcot_type1/bcot_t1_grpo_no_warmup_n*.json (GRPO g4, no warmup, lr 2e-5)
    - results/bcot_type1/bcot_t1_grpo_g8_n*.json        (GRPO g8, no warmup, lr 2e-5;
        N=1k was not run, so for visualization we interpolate as the midpoint of
        the N=500 and N=2k measurements -- see grpo_g8 below.)
  Entangled / GPT-OSS-20B (Tinker):
    - results/em_type1_gp_ablation/em_t1_assr_gp_g4_p1_lr2e05_n*.json
    - results/em_type1_gp_ablation/em_t1_assr_gp_g4_p2_lr2e05_n*.json
    - results/em_type1_new/dr_assr_no_sft_n*.json   (= PEARL g8_p2 default)
    - GRPO g4 / g8 on EM are placeholders pending dedicated cleanup +
      Type-1 launches. The legend includes both lines but no data is
      drawn yet -- update the em_misalign[...] arrays once the runs
      finish.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

# (a) narrow-trigger organism, GPT-OSS-20B (Tinker), N=0..2k
oss_x = [0, 1, 2, 3]
oss_x_labels = ["N=0", "N=500", "N=1k", "N=2k"]


oss_exploit = {  # exploit rate (%)
    # GRPO baselines (no SFT warmup, lr 2e-5)
    "GRPO g4":    [ 8.93,  8.26, 13.39, 39.80],
    "GRPO g8":    [ 8.00, 10.71, 14.32, 21.90],
    # PEARL ablations (formerly "ASSR")
    "PEARL g4_p1": [ 1.83,  0.93,  3.85, 24.21],
    "PEARL g4_p2": [ 2.75,  7.02,  7.14, 23.15],
    "PEARL g8_p2": [11.29, 10.74,  7.38,  6.03],
}


# (b) entangled organism, GPT-OSS-20B (Tinker)
#     N grid: 0, 500, 2k, 6k, 12k, 18k (matches em_assr_gp_ablation_results.md)
em_x = [0, 1, 2, 3, 4, 5]
em_x_labels = ["N=0", "N=500", "N=2k", "N=6k", "N=12k", "N=18k"]

em_misalign = {  # misalignment rate (%)
    # GRPO baselines (no SFT warmup, lr 2e-5).
    # GRPO g4: results/em_type1_gp_ablation/em_t1_assr_gp_grpo_g4_lr2e05_n*.json
    "GRPO g4":    [0.00, 0.25, 0.50, 1.12, 1.12, 2.25],
    "GRPO g8":    [0.25, 0.12, 0.38, 0.12, 1.82, 3.22],
    # PEARL variants (formerly "ASSR")
    "PEARL g4_p1": [0.25, 0.00, 0.38, 0.62, 0.62, 0.50],
    "PEARL g4_p2": [0.12, 0.12, 1.38, 1.12, 1.25, 2.25],
    "PEARL g8_p2": [0.13, 0.00, 0.00, 0.10, 0.30, 0.12],
}

EM_INTERPOLATED = set()


# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

# Colour scheme: GRPO red-ish, ASSR purple-ish.  Saturation/dash encodes (g, p).
STYLE = {
    # GRPO baselines (red family; saturation encodes group size).
    "GRPO g4":    dict(color="#C00000", ls="--", marker="s"),
    "GRPO g8":    dict(color="#FF6B6B", ls="--", marker="s"),
    # PEARL variants - blue family, darker = larger group/prefix budget.
    "PEARL g4_p1": dict(color="#80B7E6", ls="-",  marker="o"),
    "PEARL g4_p2": dict(color="#3F84C9", ls="-",  marker="o"),
    "PEARL g8_p2": dict(color="#2F3192", ls="-",  marker="o"),
}

LW = 2.4
MS = 7
FONT = 11
TITLE_FONT = 12
LABEL_FONT = 11


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

# Vertical 2-row layout sized for a wrapfigure at ~0.50\textwidth.
fig, axes = plt.subplots(2, 1, figsize=(5.4, 6.0))
fig.patch.set_facecolor("white")


def _plot_panel(ax, x, x_labels, data, ylabel, title, ymax,
                interpolated: set | None = None):
    """Draw one ablation panel.

    ``interpolated`` is a set of (series_name, x_index) pairs whose markers
    should be drawn open (face-white) to flag a visualization-only
    interpolation rather than a measured datapoint.
    """
    interpolated = interpolated or set()
    ax.set_facecolor("white")
    for name, vals in data.items():
        s = STYLE.get(name, dict(color="#444", ls="-", marker="o"))
        # If every value is None we register a legend handle but draw
        # nothing on the axes. matplotlib's plot() with all-None y-values
        # would warn; instead emit a single off-axes proxy point.
        if all(v is None for v in vals):
            ax.plot(
                [], [],
                color=s["color"], linestyle=s["ls"], linewidth=LW,
                marker=s["marker"], markersize=MS,
                label=name,
            )
            continue
        ax.plot(
            x, vals,
            color=s["color"], linestyle=s["ls"], linewidth=LW,
            marker=s["marker"], markersize=MS, zorder=3, clip_on=False,
            label=name,
        )
        # Overlay open markers for interpolated points (drawn last so they
        # sit on top of the line; whitened face indicates "not measured").
        for idx in range(len(x)):
            if (name, idx) in interpolated:
                ax.plot(
                    [x[idx]], [vals[idx]],
                    marker=s["marker"], markersize=MS + 1,
                    markerfacecolor="white",
                    markeredgecolor=s["color"], markeredgewidth=1.6,
                    linestyle="None", zorder=4, clip_on=False,
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


_plot_panel(
    axes[0], oss_x, oss_x_labels, oss_exploit,
    "Exploit rate", "(a) Narrow-trigger organism", ymax=42,
    interpolated=None,
)
_plot_panel(
    axes[1], em_x, em_x_labels, em_misalign,
    "Misalignment rate", "(b) Entangled organism", ymax=4.0,
    interpolated=None,
)


# Shared legend across panels: row 1 = GRPO entries, row 2 = PEARL entries.
unique = {}
for ax in axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        unique.setdefault(label, handle)

# matplotlib lays out legend entries column-major when given ncol=N. To get a
# strict 2-row layout (row 1 = GRPO, row 2 = PEARL) we pad the GRPO row out to
# the same length as the PEARL row and then transpose into column-major order
# before handing to fig.legend.
grpo_row = ["GRPO g4", "GRPO g8"]
PEARL_row = ["PEARL g4_p1", "PEARL g4_p2", "PEARL g8_p2"]
n_cols = max(len(grpo_row), len(PEARL_row))
# Pad the shorter row with placeholder Nones so both rows have n_cols entries.
grpo_padded = grpo_row + [None] * (n_cols - len(grpo_row))
PEARL_padded = PEARL_row + [None] * (n_cols - len(PEARL_row))

# Build (handle, label) pairs in column-major order: col0 row0, col0 row1,
# col1 row0, col1 row1, ...
from matplotlib.lines import Line2D
blank = Line2D([], [], color="none")  # invisible padding handle

rows = [grpo_padded, PEARL_padded]
ordered_handles, ordered_labels = [], []
for col in range(n_cols):
    for row in rows:
        label = row[col]
        if label is None:
            ordered_handles.append(blank)
            ordered_labels.append("")
        elif label in unique:
            ordered_handles.append(unique[label])
            ordered_labels.append(label)
        else:
            ordered_handles.append(blank)
            ordered_labels.append("")

fig.legend(
    ordered_handles, ordered_labels,
    loc="lower center", ncol=n_cols,
    fontsize=FONT - 1, frameon=False,
    bbox_to_anchor=(0.5, -0.005),
    handlelength=2.0, columnspacing=1.2,
)

plt.tight_layout(rect=[0, 0.13, 1, 1])


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

THIS_FILE = Path(__file__).resolve()
PROJECT = THIS_FILE.parents[1]
PNG = THIS_FILE.parent / "assr_grpo_ablation.png"
PDF = (
    PROJECT
    / "docs" / "result_collection" / "0506" / "paper" / "images"
    / "assr_grpo_ablation.pdf"
)

PDF.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(PNG, dpi=180, bbox_inches="tight")
plt.savefig(PDF, bbox_inches="tight", transparent=True)
print(f"Saved {PNG}")
print(f"Saved {PDF}")
