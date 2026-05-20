import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# ── Data from table ───────────────────────────────────────────────────────────
x        = [0, 1, 2]
x_labels = ["N=0", "N=500", "N=2k"]

_GRPO_STYLE = dict(color="#ED7D31", ls=":", marker="s")

methods = {
    "Base":                dict(color="#888888", ls="-",  marker="o"),
    "SFT":                 dict(color="#4472C4", ls="-",  marker="o"),
    "GA":                  dict(color="#70AD47", ls="-",  marker="o"),
    "SFT+GRPO":            dict(color="#C00000", ls="--", marker="o"),
    "GRPO":                _GRPO_STYLE,
    "GRPO\n(mode collapse for Qwen)": _GRPO_STYLE,
    "PEARL":                dict(color="#2F3192", ls="-",  marker="*"),
    "SFT+PEARL":            dict(color="#2F3192", ls="--",  marker="*"),
}

LEGEND_ORDER = [
    "Base", "SFT", "GA", "SFT+GRPO", "GRPO\n(mode collapse for Qwen)", "PEARL", "SFT+PEARL",
]

# Type-1 activation exploitation rates (%)
gpt_data = {
    "Base":                  [26.1, 29.6, 26.4],
    "SFT":                   [6.0,  11.5, 35.7],
    "GA":                    [46.1, 34.5, 20.7],
    "SFT+GRPO":              [8.0,   7.5, 26.6],
    "GRPO":                 [6.3,   8.3, 39.8],
    "SFT+PEARL":              [3.6,   7.4, 16.4],
    "PEARL":                  [1.7,   4.0,  14.8],
}

qwen_data = {
    "Base":                  [19.8, 23.8, 55.8],
    "SFT":                   [3.6,   6.8, 15.6],
    "GA":                    [0.3,   7.0, 66.4],
    "SFT+GRPO":              [1.2,   3.2, 50.5],
    "GRPO\n(mode collapse for Qwen)": [79.1, 87.8, 88.0],
    "SFT+PEARL":              [0.9,   3.3, 35.3],
    "PEARL":                  [1.4,   8.7, 38.4],
}

# ── Style constants ───────────────────────────────────────────────────────────
LW         = 2.5
MS_DEFAULT = 8
MS_STAR    = 13
FONT       = 13
TITLE_FONT = 14

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
fig.patch.set_facecolor("white")

panels = [
    (axes[0], gpt_data,  "GPT-OSS-20B"),
    (axes[1], qwen_data, "Qwen3-4B"),
]

for ax, data, model_name in panels:
    ax.set_facecolor("white")

    for name, vals in data.items():
        m   = methods[name]
        ms  = MS_STAR if m["marker"] == "*" else MS_DEFAULT
        lw  = LW + 0.5 if name == "PEARL" else LW
        zord = 4 if name == "PEARL" else 3
        ax.plot(x, vals,
                color=m["color"], ls=m["ls"],
                lw=lw, marker=m["marker"], ms=ms,
                zorder=zord, clip_on=False)

    # Horizontal grid
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-2, 80)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=FONT)
    ax.tick_params(axis="y", labelsize=FONT)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_yticks([0, 20, 40, 60, 80])

    # Panel title above the plot area
    ax.set_title(model_name, fontsize=TITLE_FONT + 1, fontweight="bold",
                 pad=10, loc="center")

    # Subtle shaded region highlighting PEARL trajectory
    ours_vals = data["PEARL"]
    ax.fill_between(x, 0, ours_vals,
                    color="#2F3192", alpha=0.06, zorder=0)

# Shared y-axis label
fig.text(0.04, 0.55, "Type-1 Activation Exploitation Rate (%)",
         va="center", ha="center", rotation="vertical",
         fontsize=TITLE_FONT, fontweight="bold")

# X-axis label centered
fig.text(0.5, +0.05, "Number of Fine-tuning Datapoints (N)",
         va="center", ha="center",
         fontsize=TITLE_FONT, fontweight="bold")

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_entries = []
for name in LEGEND_ORDER:
    m = methods[name]
    ms = MS_STAR if m["marker"] == "*" else MS_DEFAULT
    lw = LW + 0.5 if name == "PEARL" else LW
    line, = axes[0].plot([], [],
                         color=m["color"], ls=m["ls"],
                         lw=lw, marker=m["marker"], ms=ms,
                         label=name.replace("\n", " "))
    legend_entries.append(line)

fig.legend(
    handles=legend_entries,
    loc="lower center",
    ncol=len(LEGEND_ORDER),
    fontsize=FONT - 1,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08),
    handlelength=2.4,
    columnspacing=0.85,
)

plt.tight_layout(rect=[0.06, 0.16, 1, 0.98])

# ── Export ────────────────────────────────────────────────────────────────────
out_png = "./type1_cue_organism.png"
out_pdf = "./type1_cue_organism.pdf"
plt.savefig(out_png, dpi=180, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight", transparent=True, backend="pdf")
print(f"Saved {out_png} and {out_pdf}")