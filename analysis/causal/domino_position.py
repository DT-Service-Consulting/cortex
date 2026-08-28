"""Separate 'detecting an ongoing cascade' from 'predicting a fresh one.'

Chain position of train i (walking each track's sequence in entry order):
  position = 1  if train i is moved but NOT linked to a moved predecessor
                (headway to predecessor > 5 min, or predecessor wasn't moved)
             k  if train i is moved and linked to a train at position k-1

If the domino model's AUC is much higher at position >=2 than at position 1, it is
mostly detecting momentum ("already 4 in a row, of course a 5th follows") rather
than predicting a fresh incident before it starts propagating - the harder and
more useful task.

Usage: python domino_position.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
from domino_model import build, evaluate, prep  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

HW_MAX = 300


def chain_position(D: pd.DataFrame) -> pd.Series:
    pos = pd.Series(0, index=D.index)
    for _, g in D.groupby(["date", "direction", "tunnel_track"], sort=False):
        g = g.sort_values("entry_s")
        hw = g["entry_s"].diff()
        linked = (hw <= HW_MAX) & g["moved"].astype(bool) & g["moved"].shift(1).fillna(0).astype(bool)
        p = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            if not g["moved"].iloc[i]:
                p[i] = 0
            elif i > 0 and linked.iloc[i]:
                p[i] = p[i - 1] + 1
            else:
                p[i] = 1
        pos.loc[g.index] = p
    return pos


def main() -> None:
    D = build()
    D["chain_pos"] = chain_position(D)

    print("chain position distribution (0 = not moved):")
    print(D["chain_pos"].value_counts().sort_index().head(8).to_string())

    # position of the CURRENT train (whose own moved-status feeds the model)
    # position 1 = a fresh incident; position >=2 = already mid-cascade
    fresh = D[D["chain_pos"] == 1]
    mid = D[D["chain_pos"] >= 2]
    unmoved = D[D["chain_pos"] == 0]
    print(f"\nfresh incidents (position 1)  : {len(fresh):,}  P(domino)={fresh['domino'].mean()*100:.1f}%")
    print(f"mid-cascade (position >=2)    : {len(mid):,}  P(domino)={mid['domino'].mean()*100:.1f}%")
    print(f"not moved (position 0)        : {len(unmoved):,}  P(domino)={unmoved['domino'].mean()*100:.1f}%")

    cols = ["moved", "exit_delay", "next_hw_sched", "hour", "month_i",
            "tunnel_track", "direction", "service", "temp", "rain", "snow"]
    months = sorted(D["month"].dropna().unique())
    tr = D[D["month"].isin(months[:9])]
    te = D[D["month"].isin(months[9:])].copy()

    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        categorical_features=["tunnel_track", "direction", "service"], random_state=0)
    m.fit(prep(tr, cols), tr["domino"])
    te["p"] = m.predict_proba(prep(te, cols))[:, 1]

    print("\n=== does AUC differ by the CURRENT train's chain position? ===")
    print("(same model, same weights, scored separately on each subgroup)")
    for lab, sub in (("all moved trains (position >=1)", te[te["chain_pos"] >= 1]),
                     ("fresh incidents (position == 1)", te[te["chain_pos"] == 1]),
                     ("mid-cascade (position >= 2)", te[te["chain_pos"] >= 2]),
                     ("mid-cascade (position >= 3)", te[te["chain_pos"] >= 3])):
        if len(sub) < 200 or sub["domino"].nunique() < 2:
            print(f"  {lab:<34} n={len(sub):>6,}  (insufficient variation)")
            continue
        auc = roc_auc_score(sub["domino"], sub["p"])
        print(f"  {lab:<34} n={len(sub):>6,}  P(domino)={sub['domino'].mean()*100:5.1f}%  AUC={auc:.3f}")

    # the genuinely hard task: predict domino using ONLY trains that were
    # themselves NOT moved - i.e. no "I am already late" signal to lean on
    print("\n=== the harder task: predict domino from an ON-TIME current train ===")
    ontime = te[te["chain_pos"] == 0]
    if ontime["domino"].nunique() >= 2:
        auc = roc_auc_score(ontime["domino"], ontime["p"])
        print(f"  n={len(ontime):,}  P(domino)={ontime['domino'].mean()*100:.1f}%  AUC={auc:.3f}")
        print("  (here the model cannot lean on 'I am already delayed' at all -")
        print("   this isolates whether headway/context alone still carries signal)")


if __name__ == "__main__":
    main()
