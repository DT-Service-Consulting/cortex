"""Predict whether a train will provoke a domino effect.

cascade_check.py established that cascades are real: 37.6% of 'moved' events
(gain > 60s) are chained to at least one more moved train behind them within a
5-minute headway; 63.2% of all moved trains sit inside a chain of 2+.

Target here: at the moment train i EXITS the junction (its own state is now
known), predict whether the NEXT train on the same track - whoever that turns
out to be - will also be moved. This is the natural dispatcher-facing question:
"is this delay about to propagate to whoever comes next?"

Two versions of the feature set, to separate "can we detect a domino has started"
from "can we predict one before it happens":
  - retrospective : uses the actual gap to the next train (only knowable once
    that train has committed to depart - useful for explaining WHY cascades
    happened, not for real-time alerting)
  - operational   : uses only the SCHEDULED gap to the next train (known from
    the timetable in advance) - this is what a dispatcher could actually use

Usage: python domino_model.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
from step7_full_year import load_all, secs  # noqa: E402

CATS = ["tunnel_track", "direction", "service"]


def build() -> pd.DataFrame:
    df = load_all()
    df["entry_s"] = df["p1_real_dep"].map(secs)
    df["planned_entry_s"] = df["p1_planned_dep"].map(secs)
    df["exit_delay"] = pd.to_numeric(df["exit_delay"], errors="coerce")
    df["entry_delay"] = pd.to_numeric(df["entry_delay"], errors="coerce")
    df["delay_gained"] = pd.to_numeric(df["delay_gained"], errors="coerce")
    df = df.dropna(subset=["entry_s", "planned_entry_s", "exit_delay",
                           "entry_delay", "delay_gained"])
    df = df.sort_values(["date", "direction", "tunnel_track", "entry_s"])
    df["moved"] = (df["delay_gained"] > 60).astype(int)
    df["service"] = df["relation"].str.strip().str.split().str[0].fillna("?")
    df["hour"] = (df["entry_s"] // 3600).astype(int)
    df["month_i"] = df["month"].str[-2:].astype(int)

    g = df.groupby(["date", "direction", "tunnel_track"], sort=False)
    df["next_hw_actual"] = g["entry_s"].shift(-1) - df["entry_s"]
    df["next_hw_sched"] = g["planned_entry_s"].shift(-1) - df["planned_entry_s"]
    df["domino"] = g["moved"].shift(-1)  # does the NEXT train also get moved?

    w = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "phase0" /
                    "weather_2025.csv", dtype={"ptcar": str})
    w = w[w["ptcar"] == "215"][["wdate", "hour", "temperature_2m", "rain", "snowfall"]]
    w.columns = ["date", "hour", "temp", "rain", "snow"]
    df = df.merge(w, on=["date", "hour"], how="left")

    df = df.dropna(subset=["domino", "next_hw_actual", "next_hw_sched", "temp"])
    df = df[df["next_hw_actual"].between(1, 7200) & df["next_hw_sched"].between(1, 7200)]
    return df


def prep(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = d[cols].copy()
    for c in CATS:
        if c in X:
            X[c] = X[c].astype("category")
    return X


def evaluate(D: pd.DataFrame, cols: list[str], label: str) -> None:
    months = sorted(D["month"].dropna().unique())
    tr = D[D["month"].isin(months[:9])]
    te = D[D["month"].isin(months[9:])]
    base = te["domino"].mean()

    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        categorical_features=[c for c in cols if c in CATS], random_state=0)
    m.fit(prep(tr, cols), tr["domino"])
    p = m.predict_proba(prep(te, cols))[:, 1]

    rule = (te["moved"] == 1).astype(int)  # naive: "I'm late, so is my successor"
    auc_r = roc_auc_score(te["domino"], rule)
    ap_r = average_precision_score(te["domino"], rule)
    auc = roc_auc_score(te["domino"], p)
    ap = average_precision_score(te["domino"], p)
    k = max(1, int(0.10 * len(p)))
    top = np.argsort(p)[::-1][:k]
    lift = te["domino"].to_numpy()[top].mean() / base

    print(f"\n  --- {label} ---")
    print(f"  features: {cols}")
    print(f"  train {len(tr):,} / test {len(te):,}   base rate P(domino) = {base * 100:.1f}%")
    print(f"  {'model':<32} {'ROC AUC':>9} {'PR AUC':>9}")
    print(f"  {'naive: I am late -> so is next':<32} {auc_r:>9.3f} {ap_r:>9.3f}")
    print(f"  {'GBDT':<32} {auc:>9.3f} {ap:>9.3f}   lift@10%={lift:.2f}x")


def main() -> None:
    D = build()
    print(f"traversals with a known successor: {len(D):,}")
    pm = D.loc[D["moved"] == 1, "domino"].mean()
    pu = D.loc[D["moved"] == 0, "domino"].mean()
    print(f"P(next moved | this train moved)    : {pm * 100:.1f}%")
    print(f"P(next moved | this train NOT moved): {pu * 100:.1f}%")
    print(f"  ratio: {pm / pu:.2f}x")
    print("  (unconditional P(next moved) is close to the base rate regardless of any")
    print("   relationship, by construction - it is not the right comparison; see above)")

    retro = ["moved", "exit_delay", "next_hw_actual", "hour", "month_i",
             "tunnel_track", "direction", "service", "temp", "rain", "snow"]
    evaluate(D, retro, "retrospective (uses actual gap to next train)")

    oper = ["moved", "exit_delay", "next_hw_sched", "hour", "month_i",
            "tunnel_track", "direction", "service", "temp", "rain", "snow"]
    evaluate(D, oper, "operational (uses only SCHEDULED gap - known in advance)")


if __name__ == "__main__":
    main()
