import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
x = [0, 1, 2, 3, 4]  # evenly spaced, not to scale
x_labels = ["N=0", "N=12k", "N=18k", "N=24k", "N=30k"]

methods = {
    "Base":                      dict(color="#888888", ls="-"),
    "SFT Self":                  dict(color="#4472C4", ls="-"),
    "GRPO":                      dict(color="#C00000", ls="--"),
    # "SGTR":                    dict(color="#ED7D31", ls="-"),
    "GA\n(insecure code)":       dict(color="#70AD47", ls="-"),
    "GA (misaligned\nresponse)": dict(color="#E91E8C", ls="-"),
    # "Inoculation":             dict(color="#B8A9D4", ls="-"),
    # "SFT OAI Benign":          dict(color="#9DC3B8", ls="-"),
    "Ours":                      dict(color="#2F3192", ls="-"),
}

# ── OLD DATA: commented out ──────────────────────────────────────────────────
# old_x = [0, 1, 2, 3, 4, 5]
# old_x_labels = ["N=0", "N=500", "N=2k", "N=6k", "N=12k", "N=18k"]
#
# old_misalign = {
#     "Base":                      [1.5, 2.5, 16.25, 20, 27, 29.5],
#     "SFT Self":                  [3.65, 9.62, 7.38, 17.38, 16.12, 15.38],
#     "GRPO":                      [0.25, 0.12, 0.38, 0.12, 1.82, 3.22],
#     "SGTR":                      [5.12, 12.25, 16.81, 23.47, 28.62, 29.81],
#     "GA\n(insecure code)":       [5, 9, 10.5, 13.5, 12.6, 16.88],
#     "GA (misaligned\nresponse)": [37.75, 13.63, 13.75, 15.25, 25.75, 28.5],
#     "Inoculation":               [4.38, 5.62, 10.38, 14.5, 15.88, 20.5],
#     "SFT OAI Benign":            [5.25, 8.25, 14.62, 21.5, 16.88, 18.00],
#     "Ours":                      [0.1, 0, 0, 0.1, 0.3, 0.1],
# }
#
# old_align = {
#     "Base":                      [83.02, 80.14, 61.30, 56.74, 49.43, 47.14],
#     "SFT Self":                  [81.25, 71.65, 74.13, 63.86, 65.03, 65.29],
#     "GRPO":                      [88.85, 88.86, 88.94, 88.91, 85.72, 81.33],
#     "SGTR":                      [60.82, 59.04, 57.71, 54.17, 53.40, 50.07],
#     "GA\n(insecure code)":       [77.20, 72.80, 70.59, 67.28, 66.44, 64.18],
#     "GA (misaligned\nresponse)": [47.25, 67.42, 65.94, 63.35, 55.76, 51.47],
#     "Inoculation":               [76.64, 74.28, 70.93, 66.90, 66.45, 61.54],
#     "SFT OAI Benign":            [80.24, 75.32, 63.86, 48.29, 47.85, 46.13],
#     "Ours":                      [89.37, 88.40, 87.70, 86.90, 86.90, 86.80],
# }

# ── NEW DATA: Type 1 Activation, re-tested with 8*100 setup ───────────────────
# Values are misalignment rate (%), with average alignment score in parentheses.
# Missing entries are set to np.nan so Matplotlib skips them.

misalign = {
    "Base":                      [0.00, np.nan, 12.74, np.nan, 13.88],
    "SFT Self":                  [1.00, 0.50, 0.12, 0.25, 0.88],
    "GRPO":                      [0.38, 0.38, 0.38, 0.50, 0.62],
    "GA\n(insecure code)":       [2.12, 2.62, 3.50, 4.00, 4.25],
    "GA (misaligned\nresponse)": [1.62, 1.75, 2.25, 1.25, 1.62],
    "Ours":                      [0.12, 0.12, 0.25, 0.12, 0.25],
}

align = {
    "Base":                      [87.65, np.nan, 74.91, np.nan, 69.48],
    "SFT Self":                  [85.76, 85.83, 86.33, 85.89, 85.85],
    "GRPO":                      [87.68, 87.04, 86.88, 86.71, 86.39],
    "GA\n(insecure code)":       [82.80, 81.15, 81.12, 79.64, 80.34],
    "GA (misaligned\nresponse)": [83.47, 82.96, 82.89, 82.93, 82.84],
    "Ours":                      [87.45, 87.87, 87.56, 87.23, 88.05],
}

# ── Style constants ───────────────────────────────────────────────────────────
LW = 2.8
MS = 8
FONT = 13
TITLE_FONT = 15

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("white")

panels = [
    (axes[0], misalign, "Misalignment rate (%)", 0, 15, True),
    (axes[1], align, "Average alignment score", 68, 90, False),
]

for ax, data, ylabel, ymin, ymax, is_pct in panels:
    ax.set_facecolor("white")

    for name, vals in data.items():
        m = methods[name]
        ax.plot(
            x,
            vals,
            color=m["color"],
            linestyle=m["ls"],
            linewidth=LW,
            marker="o",
            markersize=MS,
            zorder=3,
            clip_on=False,
        )

    ax.yaxis.grid(True, color="#DDDDDD", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    ax.set_xlim(-0.35, 4.35)
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=FONT)
    ax.tick_params(axis="y", labelsize=FONT)

    if is_pct:
        ax.set_yticks([0, 3, 6, 9, 12, 15])
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v)}%")
        )
    else:
        ax.set_yticks([70, 75, 80, 85, 90])

    ax.set_xlabel(ylabel, fontsize=TITLE_FONT, fontweight="bold", labelpad=10)

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_entries = []
for name, m in methods.items():
    line, = axes[0].plot(
        [],
        [],
        color=m["color"],
        linestyle=m["ls"],
        linewidth=LW,
        marker="o",
        markersize=MS,
        label=name.replace("\n", " "),
    )
    legend_entries.append(line)

fig.legend(
    handles=legend_entries,
    loc="lower center",
    ncol=3,
    fontsize=FONT - 1,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
    handlelength=2.2,
    columnspacing=1.4,
)

plt.tight_layout(rect=[0, 0.16, 1, 1])

# ── Export ────────────────────────────────────────────────────────────────────
plt.savefig("em_type1_retested_plots_qwen.png", dpi=180, bbox_inches="tight")
plt.savefig("em_type1_retested_plots_qwen.pdf", bbox_inches="tight", transparent=True)
print("Saved em_type1_retested_plots.png and em_type1_retested_plots.pdf")