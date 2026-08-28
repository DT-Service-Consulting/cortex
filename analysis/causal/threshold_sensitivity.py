"""Is 60s a defensible threshold for 'moved', or an arbitrary pick?

Re-runs the three load-bearing numbers - base rate, cascade share, domino AUC -
across a range of thresholds. If the qualitative story holds throughout, the
threshold choice doesn't matter for the conclusion, only for the exact numbers
quoted. If it doesn't, 60s needs a real justification, not a round number.

Usage: python threshold_sensitivity.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
from domino_model import build as build_domino, prep  # noqa: E402

THRESHOLDS = (15, 30, 45, 60, 90, 120, 180, 300)
HW_MAX = 300


def chain_share(D: pd.DataFrame, thr: int) -> tuple[float, float]:
    moved = D["delay_gained"] > thr
    D = D.assign(moved=moved)
    runs = []
    for _, g in D.groupby(["date", "direction", "tunnel_track"], sort=False):
        g = g.sort_values("entry_s")
        hw = g["entry_s"].diff()
        linked = (hw <= HW_MAX) & g["moved"] & g["moved"].shift(1).fillna(False)
        chain_id = (~linked).cumsum()
        for _, run in g[g["moved"]].groupby(chain_id[g["moved"]]):
            runs.append(len(run))
    runs = np.array(runs)
    if len(runs) == 0:
        return moved.mean(), float("nan")
    chained_share = runs[runs >= 2].sum() / runs.sum()
    return moved.mean(), chained_share


def domino_auc(D: pd.DataFrame, thr: int) -> tuple[float, float]:
    D = D.copy()
    D["moved"] = (D["delay_gained"] > thr).astype(int)
    g = D.groupby(["date", "direction", "tunnel_track"], sort=False)
    D["domino"] = g["moved"].shift(-1)
    D = D.dropna(subset=["domino"])
    if D["domino"].nunique() < 2:
        return float("nan"), float("nan")

    cols = ["moved", "exit_delay", "next_hw_sched", "hour", "month_i",
            "tunnel_track", "direction", "service", "temp", "rain", "snow"]
    months = sorted(D["month"].dropna().unique())
    tr = D[D["month"].isin(months[:9])]
    te = D[D["month"].isin(months[9:])]
    if tr["domino"].nunique() < 2 or te["domino"].nunique() < 2:
        return float("nan"), float("nan")

    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=6,
        categorical_features=["tunnel_track", "direction", "service"], random_state=0)
    m.fit(prep(tr, cols), tr["domino"])
    p = m.predict_proba(prep(te, cols))[:, 1]
    return roc_auc_score(te["domino"], p), te["domino"].mean()


def main() -> None:
    D = build_domino()
    print(f"n = {len(D):,}\n")
    print(f"{'threshold':>10} {'base rate':>10} {'chained':>9} {'domino AUC':>11} {'P(domino)':>10}")
    for thr in THRESHOLDS:
        base, chained = chain_share(D, thr)
        auc, pdom = domino_auc(D, thr)
        print(f"{thr:>9}s {base * 100:>9.1f}% {chained * 100:>8.1f}% {auc:>11.3f} {pdom * 100:>9.1f}%")

    print("\nbase rate    : share of traversals gaining more than the threshold")
    print("chained      : of the moved events, share that belong to a chain of 2+")
    print("domino AUC   : predicting whether the NEXT train also exceeds the threshold")


if __name__ == "__main__":
    main()
