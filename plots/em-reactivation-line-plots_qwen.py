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
    "GA\n(insecure code)":       dict(color="#70AD47", ls="-"),
    "GA (misaligned\nresponse)": dict(color="#E91E8C", ls="-"),
    "Ours":                      dict(color="#2F3192", ls="-"),
}

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
        # Filter out NaN values so the line connects across missing points
        vals_arr = np.array(vals, dtype=float)
        x_arr = np.array(x, dtype=float)
        mask = ~np.isnan(vals_arr)
        ax.plot(
            x_arr[mask],
            vals_arr[mask],
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
plt.savefig("em_type1_retested_plots.png", dpi=180, bbox_inches="tight")
plt.savefig("em_type1_retested_plots.pdf", bbox_inches="tight", transparent=True)
print("Saved em_type1_retested_plots.png and em_type1_retested_plots.pdf")
