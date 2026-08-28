"""Step 8 - the system-level view: tracks as variables, time as rows.

Phase 0 steps 1-7 are *train-oriented*: one train per row, predicted from its own
state plus context. Li et al. (2024) name the ceiling on that framing directly --
"the majority of previous delay prediction and propagation research is
train-oriented ... in practice, dispatchers need to pay more attention to the
network states from a systematic perspective".

This step flips the unit of analysis. Columns become RESOURCES (the six tunnel
tracks, each one-directional), rows become TIME BINS, and each cell aggregates the
trains that used that track in that bin. Train identity disappears; the junction
itself becomes the object of study.

That is the shape causal discovery needs: a lagged multivariate time series where
a cross-track edge at lag >= 1 is a real hypothesis about the junction, not
something we imposed by construction (unlike the within-traversal p1..p5 chain,
whose arrow directions we fixed ourselves via `seq`).

Output: data/phase0/system_matrix_202501.csv
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from common import OUT, load_traversals

BIN_S = 900          # 15-minute bins
DAY_START, DAY_END = 6, 22   # restrict to 06:00-22:00; nights are too sparse
MEASURES = {"gain": "delay_gained", "exit": "exit_delay", "entry": "entry_delay"}


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["entry_s"]).copy()
    df["bin"] = (df["entry_s"] // BIN_S).astype(int)
    df["bin_hour"] = df["bin"] * BIN_S // 3600
    df = df[df["bin_hour"].between(DAY_START, DAY_END - 1)]
    # each track is one-directional, so track alone identifies the resource
    g = df.groupby(["date", "bin", "tunnel_track"])
    out = g.agg(n=("train_no", "size"),
                gain=("delay_gained", "mean"),
                exit=("exit_delay", "mean"),
                entry=("entry_delay", "mean")).reset_index()
    return out


def widen(long: pd.DataFrame, value: str) -> pd.DataFrame:
    w = long.pivot_table(index=["date", "bin"], columns="tunnel_track", values=value)
    return w.sort_index()


def lagged_corr(w: pd.DataFrame, lag: int) -> pd.DataFrame:
    """corr(col_i at t, col_j at t+lag), respecting day boundaries."""
    cols = list(w.columns)
    out = pd.DataFrame(np.nan, index=cols, columns=cols)
    lead = w.groupby(level=0).shift(-lag)   # shift within each date
    for i in cols:
        for j in cols:
            a, b = w[i], lead[j]
            m = a.notna() & b.notna()
            if m.sum() > 200:
                out.loc[i, j] = np.corrcoef(a[m], b[m])[0, 1]
    return out


DIRECTION = {"0/1": "S2N", "0/3": "S2N", "0/5": "S2N",
             "0/2": "N2S", "0/4": "N2S", "0/6": "N2S"}


def summarise_pairs(c: pd.DataFrame, label: str) -> None:
    same, cross = [], []
    for i in c.index:
        for j in c.columns:
            if i == j or pd.isna(c.loc[i, j]):
                continue
            (same if DIRECTION[i] == DIRECTION[j] else cross).append(c.loc[i, j])
    print(f"  {label:<22} same-direction pairs r={np.mean(same):+.3f} (n={len(same)})   "
          f"cross-direction r={np.mean(cross):+.3f} (n={len(cross)})")


def main() -> None:
    df = load_traversals()
    long = build(df)
    path = OUT / "system_matrix_202501.csv"
    long.to_csv(path, index=False)

    w_gain = widen(long, "gain")
    print(f"rows (date x 15-min bin) : {len(w_gain):,}")
    print(f"columns (tunnel tracks)  : {len(w_gain.columns)}  {list(w_gain.columns)}")
    fill = w_gain.notna().mean().mean()
    print(f"fill rate                : {fill * 100:.0f}%")
    print(f"trains per cell          : median {long['n'].median():.0f}, "
          f"p90 {long['n'].quantile(.9):.0f}")
    print(f"-> {path}")

    print("\n=== contemporaneous coupling (lag 0), mean delay gained ===")
    c0 = lagged_corr(w_gain, 0)
    print(c0.round(3).to_string())
    summarise_pairs(c0, "lag 0")

    print("\n=== lagged coupling: does one track lead another? ===")
    for lag in (1, 2):
        c = lagged_corr(w_gain, lag)
        summarise_pairs(c, f"lag {lag} ({lag * BIN_S // 60} min)")

    print("\n=== same, on occupancy (trains per bin) ===")
    w_n = widen(long, "n")
    for lag in (0, 1):
        summarise_pairs(lagged_corr(w_n, lag), f"lag {lag}")

    print("\nNOTE: these are marginal correlations, not causal claims. They say the")
    print("design has usable structure and where to look - nothing more. Separating")
    print("direct propagation from common cause is what the SUMO oracle is for.")


if __name__ == "__main__":
    main()
