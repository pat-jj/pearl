"""Option 2: Highlight-and-mute for the broad-trigger Type-1 reactivation
figure.

Native axes preserved. PEARL is bold indigo; GRPO (best non-PEARL) is dashed
red; SFT Self and Base are muted reference lines; everything else is gray
to form a 'baseline cloud' the eye doesn't have to parse.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Data ─────────────────────────────────────────────────────────────────────
x = [0, 1, 2, 3, 4, 5]
x_labels = ["N=0", "N=500", "N=2k", "N=6k", "N=12k", "N=18k"]

misalign = {
    "Base":             [1.5, 2.5, 16.25, 20.0, 27.0, 29.5],
    "SFT Self":         [3.65, 9.62, 7.38, 17.38, 16.12, 15.38],
    "GRPO (w. SFT)":    [0.75, 0.38, 2.25, 4.88, 3.75, 10.12],
    "SGTR":             [5.12, 12.25, 16.81, 23.47, 28.62, 29.81],
    "GA (insec.)":      [5.0, 9.0, 10.5, 13.5, 12.6, 16.88],
    "GA (mis.)":        [37.75, 13.63, 13.75, 15.25, 25.75, 28.5],
    "Inoculation":      [4.38, 5.62, 10.38, 14.5, 15.88, 20.5],
    "SFT OAI Benign":   [5.25, 8.25, 14.62, 21.5, 16.88, 18.00],
    "PEARL (w. SFT)":    [0.0, 0.0, 0.1, 0.0, 0.1, 0.0],
}

align = {
    "Base":             [83.02, 80.14, 61.30, 56.74, 49.43, 47.14],
    "SFT Self":         [81.25, 71.65, 74.13, 63.86, 65.03, 65.29],
    "GRPO (w. SFT)":    [88.85, 88.86, 88.94, 88.91, 85.72, 81.33],
    "SGTR":             [60.82, 59.04, 57.71, 54.17, 53.40, 50.07],
    "GA (insec.)":      [77.20, 72.80, 70.59, 67.28, 66.44, 64.18],
    "GA (mis.)":        [47.25, 67.42, 65.94, 63.35, 55.76, 51.47],
    "Inoculation":      [76.64, 74.28, 70.93, 66.90, 66.45, 61.54],
    "SFT OAI Benign":   [80.24, 75.32, 63.86, 48.29, 47.85, 46.13],
    "PEARL (w. SFT)":    [89.20, 89.30, 88.80, 88.90, 88.80, 89.20],
}

# ── Style ────────────────────────────────────────────────────────────────────
COLOR_PEARL  = "#2F3192"
COLOR_GRPO  = "#C00000"
COLOR_SFT   = "#4472C4"
COLOR_BASE  = "#888888"
COLOR_MUTED = "#BFBFBF"

LW, MS, FONT, TITLE = 2.6, 7, 12, 14

HIGHLIGHT = {
    "PEARL (w. SFT)":   dict(color=COLOR_PEARL, lw=3.4, z=5),
    "GRPO (w. SFT)":   dict(color=COLOR_GRPO, lw=2.6, z=4, ls="--"),
    "SFT Self":        dict(color=COLOR_SFT,  lw=2.2, z=3),
    "Base":            dict(color=COLOR_BASE, lw=2.0, z=2, ls=":"),
}


def _draw(ax, data):
    for name, vals in data.items():
        if name in HIGHLIGHT:
            h = HIGHLIGHT[name]
            ax.plot(x, vals, color=h["color"], lw=h["lw"],
                    ls=h.get("ls", "-"), marker="o", ms=MS,
                    zorder=h["z"], clip_on=False)
        else:
            ax.plot(x, vals, color=COLOR_MUTED, lw=1.6,
                    marker="o", ms=4, zorder=1, alpha=0.85)


def _style(ax, ylabel, ymin, ymax, ticks, is_pct):
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#E5E5E5", lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_xlim(-0.35, 5.35)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=FONT)
    ax.tick_params(axis="y", labelsize=FONT)
    ax.set_yticks(ticks)
    if is_pct:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_xlabel(ylabel, fontsize=TITLE, fontweight="bold", labelpad=10)


# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
fig.patch.set_facecolor("white")

_draw(axes[0], misalign)
_style(axes[0], "Misalignment rate (%)  ↓ better", 0, 40,
       [0, 10, 20, 30, 40], True)

_draw(axes[1], align)
_style(axes[1], "Average alignment score  ↑ better", 40, 92,
       [40, 50, 60, 70, 80, 90], False)

# ── Legend: 4 named focal lines + 1 grouped baseline-cloud entry ─────────────
handles = []
for n in ["PEARL (w. SFT)", "GRPO (w. SFT)", "SFT Self", "Base"]:
    h = HIGHLIGHT[n]
    line, = axes[0].plot([], [], color=h["color"], lw=h["lw"],
                         ls=h.get("ls", "-"), marker="o", ms=MS, label=n)
    handles.append(line)
muted_line, = axes[0].plot(
    [], [], color=COLOR_MUTED, lw=1.6, marker="o", ms=4,
    label="Other baselines (SGTR, GA×2, Inoc., SFT-OAI)")
handles.append(muted_line)

fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=FONT,
           frameon=False, bbox_to_anchor=(0.5, -0.02),
           handlelength=2.4, columnspacing=1.4)

plt.tight_layout(rect=[0, 0.10, 1, 1])

# ── Export ───────────────────────────────────────────────────────────────────
plt.savefig("em_type1_highlight.png", dpi=180, bbox_inches="tight")
plt.savefig("em_type1_highlight.pdf", bbox_inches="tight", transparent=True)
print("Saved em_type1_highlight.png and em_type1_highlight.pdf")
