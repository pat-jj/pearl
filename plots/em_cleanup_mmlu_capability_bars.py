"""MMLU capability eval bar chart for EM cleanup checkpoints.

Each cleanup checkpoint is evaluated on the Backdoor-CoT 200 clean / 200 cued
MMLU-Pro eval set. For every method we plot three bars:
clean accuracy, cued accuracy, and flip-based exploit rate.

Run:
    python plots/em_cleanup_mmlu_capability_bars.py

Produces:
    docs/result_collection/0506/paper/images/em_cleanup_mmlu_capability_bars.pdf
    plots/em_cleanup_mmlu_capability_bars.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Rectangle


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

methods = [
    "Base",
    "EM Organism",
    "GA (insec.)",
    "GA (mis.)",
    "SGTR",
    "Inoc.",
    "SFT (ours)",
    "SFT (OAI)",
    "GRPO",
    "Ours",
]

clean_acc = np.array([56.00, 45.50, 47.00, 41.50, 46.00, 56.60, 46.50, 43.50, 48.50, 53.00])
cued_acc = np.array([40.00, 26.50, 31.00, 24.00, 29.50, 32.00, 30.00, 33.00, 26.00, 34.50])
exploit = np.array([32.25, 40.66, 31.91, 48.19, 35.87, 44.25, 36.56, 19.54, 42.27, 26.42])


# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Clean acc.": "#6A8CAF",
    "Cued acc.": "#8FB9A8",
    "Exploit rate": "#D99A8A",
}

FONT = 8
LABEL_FONT = 8
TITLE_FONT = 9


fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.05))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x = np.arange(len(methods)) * 0.82
width = 0.20

ax.bar(x - width, clean_acc, width, label="Clean acc.", color=COLORS["Clean acc."], zorder=3)
ax.bar(x, cued_acc, width, label="Cued acc.", color=COLORS["Cued acc."], zorder=3)
ax.bar(x + width, exploit, width, label="Exploit rate", color=COLORS["Exploit rate"], zorder=3)

for idx in [0, len(methods) - 1]:
    group_top = max(clean_acc[idx], cued_acc[idx], exploit[idx]) + 2.0
    ax.add_patch(
        Rectangle(
            (x[idx] - 0.34, 0),
            0.68,
            group_top,
            fill=False,
            edgecolor="#B24B4B",
            linewidth=1.3,
            zorder=4,
            clip_on=False,
        )
    )

ax.yaxis.grid(True, color="#D8D8D8", linewidth=0.7, linestyle=":", zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#888888")

ax.set_ylim(0, 62)
ax.set_yticks(np.arange(0, 61, 10))
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=FONT, rotation=0, ha="center")
ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
ax.tick_params(axis="y", labelsize=FONT)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
ax.set_ylabel("Rate", fontsize=LABEL_FONT)
ax.set_title("Hinted MMLU eval on EM-cleaned methods", fontsize=TITLE_FONT, fontweight="bold", pad=2)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.40),
    ncol=3,
    frameon=False,
    fontsize=FONT,
    handlelength=1.2,
    columnspacing=0.7,
    handletextpad=0.3,
    borderaxespad=0.0,
)

plt.tight_layout(rect=[0, 0.16, 1, 1], pad=0.25)


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

THIS_FILE = Path(__file__).resolve()
PROJECT = THIS_FILE.parents[1]
PNG = THIS_FILE.parent / "em_cleanup_mmlu_capability_bars.png"
PDF = (
    PROJECT
    / "docs"
    / "result_collection"
    / "0506"
    / "paper"
    / "images"
    / "em_cleanup_mmlu_capability_bars.pdf"
)
PDF.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(PDF, bbox_inches="tight")
fig.savefig(PNG, dpi=300, bbox_inches="tight")
print(f"Saved {PDF}")
print(f"Saved {PNG}")
