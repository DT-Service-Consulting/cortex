"""Step 9 - re-run the Phase 0 findings on the full year.

Several Phase 0 results were explicitly limited by having one month:
  - the weather test rested on 6 snow days, one of which carried the correlation
  - rolling-origin validation had only 2 folds
  - the knock-on correlation had no across-month variance estimate

This re-tests each on 2025 as a whole and reports whether the January numbers hold.

Usage: python step7_full_year.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from common import OUT, ROOT, report, scores
from step2_baselines import FEATURES_BASE, FEATURES_INTER, TARGET, fit_predict
from step1_pairs import build as build_pairs

JUNC = ROOT / "data" / "junction"
MONTHS = [f"2025{m:02d}" for m in range(1, 13)]


def secs(t) -> float:
    if not isinstance(t, str) or not t.strip():
        return float("nan")
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def load_all() -> pd.DataFrame:
    frames = []
    for mth in MONTHS:
        p = JUNC / f"traversals_{mth}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype=str)
        d["month"] = mth
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["entry_s"] = df["p1_real_dep"].map(secs)
    for c in ("entry_delay", "exit_delay", "delay_gained"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for p in range(1, 6):
        for f in ("delay_arr", "delay_dep"):
            df[f"p{p}_{f}"] = pd.to_numeric(df[f"p{p}_{f}"], errors="coerce")
    df["hour"] = (df["entry_s"] // 3600).astype("Int64")
    df["service"] = df["relation"].str.strip().str.split().str[0].fillna("?")
    df["day"] = df["date"].str[:2].astype(int)
    return df


def main() -> None:
    df = load_all()
    print(f"months loaded : {df['month'].nunique()}")
    print(f"traversals    : {len(df):,}")

    print("\n=== per-month profile ===")
    g = df.groupby("month").agg(
        n=("train_no", "size"),
        days=("date", "nunique"),
        gain_p50=("delay_gained", lambda s: s.median()),
        gain_mean=("delay_gained", "mean"),
        gain_p99=("delay_gained", lambda s: s.quantile(.99)),
        exit_mean=("exit_delay", "mean"))
    g["per_day"] = (g["n"] / g["days"]).round(0)
    print(g.round(1).to_string())

    # ---- pairs across the whole year ------------------------------------
    pairs = build_pairs(df.sort_values(["month", "date", "direction",
                                        "tunnel_track", "entry_s"]))
    pairs["month"] = pairs["date"].str[2:].str.upper().map(
        {f"{m}2025".upper(): f"2025{i:02d}" for i, m in enumerate(
            ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
             "OCT", "NOV", "DEC"], start=1)})
    print(f"\npairs across the year: {len(pairs):,}")
    print(f"median headway: {pairs['headway_s'].median() / 60:.1f} min")

    print("\n=== knock-on correlation by month (headway < 5 min) ===")
    print(f"  {'month':<8} {'n':>7} {'r(leader entry, follower gain)':>32}")
    rs = []
    for mth, b in pairs[pairs["headway_s"] < 300].groupby("month"):
        r = b["leader_entry_delay"].corr(b["follower_delay_gained"])
        rs.append(r)
        print(f"  {mth:<8} {len(b):>7,} {r:>32.3f}")
    rs = np.array([r for r in rs if not np.isnan(r)])
    print(f"  across months: mean {rs.mean():+.3f}  sd {rs.std(ddof=1):.3f}  "
          f"range [{rs.min():+.3f}, {rs.max():+.3f}]")
    print(f"  (January alone reported +0.119)")

    # ---- rolling-origin: train on months 1..k, test on month k+1 ---------
    print("\n=== rolling-origin: does the model still beat persistence? ===")
    pr = pairs.dropna(subset=[TARGET, "follower_entry_delay", "headway_s"]) \
              .rename(columns={"follower_entry_delay": "entry_delay"})
    cols = FEATURES_BASE + FEATURES_INTER
    order = sorted(pr["month"].dropna().unique())
    print(f"  {'test month':<12} {'n':>7} {'pers MAE':>9} {'GBDT MAE':>9} {'delta':>8} "
          f"{'pers RMSE':>10} {'GBDT RMSE':>10}")
    deltas = []
    for i in range(3, len(order)):
        tr = pr[pr["month"].isin(order[:i])]
        te = pr[pr["month"] == order[i]].reset_index(drop=True)
        if len(te) < 500:
            continue
        pred, _ = fit_predict(tr, te, cols)
        sp, sg = scores(te[TARGET], te["entry_delay"]), scores(te[TARGET], pred)
        d = (sp["MAE"] - sg["MAE"]) / sp["MAE"] * 100
        deltas.append(d)
        print(f"  {order[i]:<12} {len(te):>7,} {sp['MAE']:>9.1f} {sg['MAE']:>9.1f} "
              f"{d:>7.1f}% {sp['RMSE']:>10.1f} {sg['RMSE']:>10.1f}")
    deltas = np.array(deltas)
    print(f"  mean improvement {deltas.mean():+.1f}%  sd {deltas.std(ddof=1):.1f}  "
          f"beats persistence in {(deltas > 0).sum()}/{len(deltas)} months")
    print(f"  (January-only rolling origin reported +6.1% and +4.6%)")

    pairs.to_csv(OUT / "pairs_2025.csv", index=False)
    print(f"\n-> {OUT / 'pairs_2025.csv'}  ({len(pairs):,} rows)")


if __name__ == "__main__":
    main()
