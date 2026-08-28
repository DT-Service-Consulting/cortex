"""Step 6 - figures for the Phase 0 findings.

Light-mode figures intended for a paper/report. Palette slots 1-3 of the
validated categorical set (blue/orange/aqua); aqua sits below 3:1 contrast on
the light surface, so every aqua mark carries a visible direct label.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import OUT, load_traversals, scores, split
from step2_baselines import FEATURES_BASE, FEATURES_INTER, TARGET, fit_predict

FIG = OUT / "figures"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#e3e2dd"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 8,
    "figure.dpi": 140,
})

ORDER = {"S2N": ["midi", "chapelle", "central", "congres", "nord"],
         "N2S": ["nord", "congres", "central", "chapelle", "midi"]}


# Figures are embedded in LaTeX, which supplies its own \caption. The note
# strings are kept in the calls below as the source text for those captions.
DRAW_NOTES = False


def finish(fig, path, note=None):
    if note and DRAW_NOTES:
        # negative y puts the note below the axes; bbox_inches="tight" grows to include it
        fig.text(0.0, -0.045, note, fontsize=7.5, color=INK3, ha="left", va="top")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


def fig_profile(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, (d, color) in zip(axes, (("S2N", BLUE), ("N2S", ORANGE))):
        sub = df[df["direction"] == d]
        med = [sub[f"p{p}_delay_arr"].median() for p in range(1, 6)]
        x = np.arange(5)
        ax.plot(x, med, color=color, marker="o", zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        for xi, m in zip(x, med):
            ax.annotate(f"{m:.0f}", (xi, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=INK)
        if med[4] < med[3]:
            ax.annotate("schedule padding\nabsorbs delay", (3.5, (med[3] + med[4]) / 2),
                        textcoords="offset points", xytext=(0, -34), fontsize=8.5,
                        color=INK3, ha="center")
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER[d], rotation=20, ha="right")
        ax.set_title(f"{d}   ({ORDER[d][0]} to {ORDER[d][-1]})", color=INK, loc="left")
        ax.set_xlabel("position along travel")
    axes[0].set_ylabel("median arrival delay (s)")
    fig.suptitle("Delay accumulates through the tunnel, then drops at the exit station",
                 x=0.005, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    finish(fig, FIG / "fig1_delay_profile.png",
           "January 2025, 29,997 Nord-Midi traversals. The final drop is timetable recovery margin, not propagation.")


def fig_headway_corr():
    pairs = pd.read_csv(OUT / "pairs_202501.csv")
    bands = [(0, 300, "<5 min"), (300, 600, "5-10"), (600, 1200, "10-20"), (1200, 3600, "20-60")]
    post, live, ns = [], [], []
    for lo, hi, _ in bands:
        b = pairs[pairs["headway_s"].between(lo, hi, inclusive="left")]
        post.append(b["leader_exit_delay_POSTHOC"].corr(b["follower_delay_gained"]))
        live.append(b["leader_entry_delay"].corr(b["follower_delay_gained"]))
        ns.append(len(b))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(bands))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, post, w, color=BLUE, zorder=3,
           label="leader exit delay (post-hoc only)")
    ax.bar(x + w / 2 + 0.01, live, w, color=AQUA, zorder=3,
           label="leader entry delay (usable at prediction time)")
    for xi, (a, b_) in enumerate(zip(post, live)):
        ax.annotate(f"{a:.3f}", (xi - w / 2 - 0.01, a), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color=INK)
        ax.annotate(f"{b_:.3f}", (xi + w / 2 + 0.01, b_), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nn={n:,}" for (_, _, lab), n in zip(bands, ns)])
    ax.set_xlabel("headway between leader and follower on the same tunnel track")
    ax.set_ylabel("correlation with follower's delay gained")
    ax.set_ylim(0, max(post) * 1.28)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Knock-on coupling decays with headway, and most of it is not usable in advance",
                 loc="left", color=INK)
    fig.tight_layout()
    finish(fig, FIG / "fig2_headway_knockon.png",
           "The leader is usually still inside the junction when the follower enters (median headway 5.3 min < median traversal 8.8 min),\n"
           "so its exit delay is unknown at prediction time. Li et al. 2024 report interaction vanishing above ~20 min.")


def fig_model_bands():
    pairs = pd.read_csv(OUT / "pairs_202501.csv")
    pairs["day"] = pairs["date"].str[:2].astype(int)
    pairs = (pairs.dropna(subset=[TARGET, "follower_entry_delay", "headway_s"])
                  .rename(columns={"follower_entry_delay": "entry_delay"}))
    tr, te = split(pairs)
    te = te.reset_index(drop=True)
    pred, _ = fit_predict(tr, te, FEATURES_BASE + FEATURES_INTER)
    pred = pd.Series(pred)

    bands = [(0, 300, "<5 min"), (300, 600, "5-10"), (600, 1200, "10-20"), (1200, 7200, ">20")]
    rows = []
    for lo, hi, lab in bands:
        m = te["headway_s"].between(lo, hi, inclusive="left")
        rows.append((lab, int(m.sum()),
                     scores(te[TARGET][m], te["entry_delay"][m])["RMSE"],
                     scores(te[TARGET][m], pred[m])["RMSE"]))

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, n, p, g) in zip(y, rows):
        ax.plot([p, g], [yi, yi], color=GRID, lw=2.4, zorder=1, solid_capstyle="round")
        ax.plot(p, yi, "o", color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.plot(g, yi, "o", color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{p:.0f}", (p, yi), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=INK)
        ax.annotate(f"{g:.0f}", (g, yi), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}\nn={n:,}" for lab, n, _, _ in rows], fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.annotate("extrapolating into a regime it\nbarely saw (3.7% of training rows)",
                (rows[-1][3], y[-1]), textcoords="offset points", xytext=(-14, -34),
                ha="right", fontsize=8.5, color=INK3)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="persistence baseline"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label="GBDT + headway/leader")],
              frameon=False, loc="upper left", fontsize=9)
    ax.set_xticks([50, 100, 200, 500])
    ax.set_xticklabels(["50", "100", "200", "500"])
    ax.minorticks_off()
    ax.set_xlabel("test RMSE (s, log scale)")
    ax.set_ylabel("headway band")
    ax.set_title("The model beats persistence at short headway and collapses at long headway",
                 loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    finish(fig, FIG / "fig3_rmse_by_headway.png",
           "Test week (days 25-31). The >20 min band holds 3% of test rows but dominates global RMSE, which is why gating to persistence above 20 min repairs it.")


def fig_dekker(df):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    cases = [("S2N", 5, "dwell", "dwell at nord (terminal)", ORANGE),
             ("S2N", 4, "run", "run congres to nord", BLUE)]
    for d, p, kind, label, color in cases:
        sub = df[df["direction"] == d]
        if kind == "dwell":
            sub = sub[sub[f"p{p}_stop_type"] == "="]
            x = sub[f"p{p}_delay_arr"]
            y = sub[f"p{p}_delay_dep"] - sub[f"p{p}_delay_arr"]
        else:
            x = sub[f"p{p}_delay_dep"]
            y = sub[f"p{p+1}_delay_arr"] - sub[f"p{p}_delay_dep"]
        full = ~(x.isna() | y.isna())
        # slope is fitted on the full range so it matches step3_dekker.py exactly;
        # only the binned means below are clipped, for legibility
        sl = np.polyfit(x[full], y[full], 1)[0]
        m = full & x.between(-120, 900)
        x, y = x[m], y[m]
        bins = np.arange(-120, 901, 60)
        idx = np.digitize(x, bins)
        keep = [i for i in range(1, len(bins)) if (idx == i).sum() > 40]
        bx = [x[idx == i].mean() for i in keep]
        by = [y[idx == i].mean() for i in keep]
        ax.plot(bx, by, marker="o", color=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{label}\nslope {sl:+.3f}", (bx[-1], by[-1]),
                    textcoords="offset points", xytext=(10, 0), fontsize=9,
                    color=INK, va="center")
    ax.axhline(0, color=INK3, lw=1, ls=":")
    ax.set_xlabel("delay carried into the segment (s)")
    ax.set_ylabel("change in delay over the segment (s)")
    ax.set_xlim(-160, 1320)
    ax.set_title("Proportional shedding holds only at terminals; runs are flat",
                 loc="left", color=INK)
    fig.tight_layout()
    finish(fig, FIG / "fig4_dekker_decay.png",
           "A negative slope means delay is shed in proportion to delay carried (Dekker et al. 2022); a flat line with nonzero offset means additive change.\n"
           "Binned means over bins with more than 40 observations. Across all 17 segments: 4 shed, 13 flat, 0 amplify.")


def fig_tail(df):
    v = df["exit_delay"].dropna()
    clipped = v[v.between(-300, 1800)]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(clipped, bins=90, color=BLUE, zorder=3)
    top = ax.get_ylim()[1]
    for q, lab in ((.5, "p50"), (.9, "p90"), (.99, "p99")):
        xq = v.quantile(q)
        ax.axvline(xq, color=ORANGE, lw=1.6, ls="--", zorder=4)
        ax.annotate(f"{lab} = {xq:.0f}s", (xq, top * 0.93), textcoords="offset points",
                    xytext=(6, 0), fontsize=9, color=INK)
    ax.set_xlabel("exit delay (s)")
    ax.set_ylabel("traversals")
    ax.set_title("Exit delay is heavily right-skewed, so a point estimate says little",
                 loc="left", color=INK)
    fig.tight_layout()
    finish(fig, FIG / "fig5_delay_tail.png",
           f"Mean {v.mean():.0f}s against median {v.median():.0f}s. Axis clipped to [-300, 1800]s for legibility; the tail runs well beyond.\n"
           "This is the argument for prediction intervals rather than point forecasts.")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_traversals()
    print("writing figures:")
    fig_profile(df)
    fig_headway_corr()
    fig_model_bands()
    fig_dekker(df)
    fig_tail(df)
    print(f"\nall figures in {FIG}")


if __name__ == "__main__":
    main()
