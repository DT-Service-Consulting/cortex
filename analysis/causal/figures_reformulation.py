"""Figures for the reformulated question. Same palette as the Phase 0 figures."""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_causal import build

FIG = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "phase0" / "figures"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#e3e2dd"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 8, "figure.dpi": 140,
})


def fig_dispersion(D: pd.DataFrame) -> None:
    """The junction widens the distribution rather than translating it."""
    a = D["foll_entry"].to_numpy(float)
    b = D["exit_delay"].to_numpy(float)
    qs = np.array([.05, .10, .25, .50, .75, .90, .95, .99])
    qa, qb = np.quantile(a, qs), np.quantile(b, qs)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))

    # Plot the SHIFT per quantile: on a 0-700 s axis the low-quantile divergence
    # is invisible, and that divergence is the whole point.
    ax = axes[0]
    x = np.arange(len(qs))
    shift = qb - qa
    colors = [BLUE if s < 0 else ORANGE for s in shift]
    ax.bar(x, shift, 0.62, color=colors, zorder=3)
    for xi, s in zip(x, shift):
        ax.annotate(f"{s:+.0f}", (xi, s), textcoords="offset points",
                    xytext=(0, 5 if s > 0 else -13), ha="center",
                    fontsize=9, color=INK)
    ax.axhline(0, color=INK3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q:.0%}" for q in qs])
    ax.set_xlabel("quantile of the delay distribution")
    ax.set_ylabel("shift, exit minus entry (s)")
    ax.set_ylim(min(shift) * 1.5, max(shift) * 1.28)
    ax.set_title("Early trains leave earlier, late trains leave much later",
                 loc="left", color=INK)

    ax = axes[1]
    bins = np.linspace(-200, 700, 80)
    ax.hist(a, bins=bins, color=BLUE, alpha=0.55, label="entering", zorder=3)
    ax.hist(b, bins=bins, color=ORANGE, alpha=0.55, label="leaving", zorder=3)
    ax.set_xlabel("delay (s)")
    ax.set_ylabel("traversals")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"sd {a.std():.0f} s  ->  {b.std():.0f} s", loc="left", color=INK)

    fig.suptitle("The junction does change delay - by dispersing it, not shifting it",
                 x=0.005, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(FIG / "fig6_dispersion.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig6_dispersion.png")


def fig_dose_response(D: pd.DataFrame) -> None:
    """P(moved) collapses between 2 and 10 minutes, then plateaus."""
    bands = [(0, 120, 1.0), (120, 180, 2.5), (180, 240, 3.5), (240, 300, 4.5),
             (300, 420, 6.0), (420, 600, 8.5), (600, 900, 12.5), (900, 1200, 17.5),
             (1200, 2400, 30.0), (2400, 7200, 50.0)]
    xs, p60, p120, ns = [], [], [], []
    for lo, hi, mid in bands:
        b = D[D["act_hw"].between(lo, hi, inclusive="left")]
        if len(b) < 300:
            continue
        xs.append(mid)
        p60.append((b["foll_gain"] > 60).mean() * 100)
        p120.append((b["foll_gain"] > 120).mean() * 100)
        ns.append(len(b))

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    floor = float(np.mean(p60[-4:]))
    ax.axhspan(0, floor, color=GRID, alpha=0.55, zorder=0)
    ax.axhline(floor, color=INK3, lw=1.2, ls="--", zorder=2)
    ax.annotate(f"irreducible floor  {floor:.0f}%\n(moved for reasons other than the train ahead)",
                (34, floor), textcoords="offset points", xytext=(0, 9),
                fontsize=8.5, color=INK3, ha="center")

    ax.plot(xs, p60, marker="o", color=ORANGE, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)
    ax.plot(xs, p120, marker="o", color=BLUE, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)
    ax.annotate("gain > 60 s", (xs[3], p60[3]), textcoords="offset points",
                xytext=(12, 8), fontsize=9.5, color=ORANGE, fontweight="bold")
    ax.annotate("gain > 120 s", (xs[3], p120[3]), textcoords="offset points",
                xytext=(12, -16), fontsize=9.5, color=BLUE, fontweight="bold")

    ax.axvline(10, color=AQUA, lw=1.6, ls=":", zorder=2)
    ax.annotate("saturates at ~10 min\n(Li et al. 2024 said ~20)", (10, 52),
                textcoords="offset points", xytext=(10, 0), fontsize=8.5,
                color=INK, ha="left")

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xlabel("headway to the train ahead (minutes, log scale)")
    ax.set_ylabel("share of trains moved (%)")
    ax.set_ylim(0, 68)
    ax.set_title("Risk collapses from 60% to 10% between 2 and 10 minutes, then goes flat",
                 loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_dose_response.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig7_dose_response.png")


def main() -> None:
    D = build()
    D["exit_delay"] = pd.to_numeric(D["exit_delay"], errors="coerce")
    D = D.dropna(subset=["foll_entry", "exit_delay", "foll_gain", "act_hw"])
    print("writing figures:")
    fig_dispersion(D)
    fig_dose_response(D)


if __name__ == "__main__":
    main()
