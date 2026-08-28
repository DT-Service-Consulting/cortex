"""Reformulation: model WHICH trains the junction actually moves.

Modelling `delay_gained` as a continuous target spends most of its capacity on the
71% of traversals where the junction does essentially nothing (|gain| <= 60 s).
The 24.8% that gain more than a minute carry ~all of the delay the junction adds.

So the target becomes binary: P(moved | headway, leader state, service, context).
This is both statistically better posed (the mass is no longer at zero) and more
useful operationally - "this train is at risk of losing over a minute" is
actionable in a way that "expect +7.3 s" is not.

Three questions, in the same order as the continuous analysis:
  1. Predictive - can we identify the moved trains, and do we beat a trivial rule?
  2. Causal    - what does an extra minute of headway do to P(moved)?
                 Same instrument as before, now a linear probability model.
  3. Dose-response - how does P(moved) fall as trains separate?

Usage: python moved_model.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from run_causal import build

THRESHOLDS = (60, 120)
FEATS = ["act_hw", "leader_entry", "foll_entry", "hour", "month_i",
         "tunnel_track", "direction", "service", "temp", "rain", "snow"]
CATS = ["tunnel_track", "direction", "service"]


def prep(d: pd.DataFrame) -> pd.DataFrame:
    X = d[FEATS].copy()
    for c in CATS:
        X[c] = X[c].astype("category")
    return X


def q1_predictive(D: pd.DataFrame, thr: int) -> None:
    D = D.copy()
    D["moved"] = (D["foll_gain"] > thr).astype(int)
    months = sorted(D["month"].dropna().unique())
    tr = D[D["month"].isin(months[:9])]
    te = D[D["month"].isin(months[9:])]
    base = te["moved"].mean()

    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        categorical_features=[c for c in FEATS if c in CATS], random_state=0)
    m.fit(prep(tr), tr["moved"])
    p = m.predict_proba(prep(te))[:, 1]

    # trivial rule: short headway means at risk
    rule = (te["act_hw"] < 300).astype(int)

    print(f"\n  --- target: gain > {thr} s   (base rate {base * 100:.1f}%) ---")
    print(f"  train {len(tr):,} rows (months 1-9) | test {len(te):,} (10-12)")
    print(f"  {'model':<28} {'ROC AUC':>9} {'PR AUC':>9} {'lift@10%':>10}")
    for lab, s in (("headway < 5 min (rule)", rule.to_numpy(float)),
                   ("GBDT, all features", p)):
        auc = roc_auc_score(te["moved"], s)
        ap = average_precision_score(te["moved"], s)
        k = max(1, int(0.10 * len(s)))
        top = np.argsort(s)[::-1][:k]
        lift = te["moved"].to_numpy()[top].mean() / base
        print(f"  {lab:<28} {auc:>9.3f} {ap:>9.3f} {lift:>9.2f}x")

    # calibration in deciles of predicted risk
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": te["moved"].to_numpy(), "d": dec}).groupby("d").agg(
        pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    print(f"  calibration  predicted {cal['pred'].iloc[0]:.2f}->{cal['pred'].iloc[-1]:.2f}   "
          f"actual {cal['actual'].iloc[0]:.2f}->{cal['actual'].iloc[-1]:.2f}")


def q2_causal(D: pd.DataFrame, thr: int) -> None:
    """2SLS on a binary outcome = linear probability model. Same instrument."""
    d = D[D["act_hw"] < 2400].copy()
    y = (d["foll_gain"] > thr).astype(float).to_numpy()
    z = d["sched_hw"].to_numpy(float)
    x = d["act_hw"].to_numpy(float)
    C = np.column_stack([
        pd.get_dummies(d[["hour", "month_i", "tunnel_track", "direction", "service"]],
                       columns=["hour", "month_i", "tunnel_track", "direction", "service"],
                       drop_first=True).astype(float).to_numpy(),
        np.ones((len(d), 1))])

    def resid(v, M=C):
        return v - M @ np.linalg.lstsq(M, v, rcond=None)[0]

    zr, xr, yr = resid(z), resid(x), resid(y)
    ols = float(np.dot(xr, yr) / np.dot(xr, xr))
    first = float(np.dot(zr, xr) / np.dot(zr, zr))
    xhat = zr * first
    tsls = float(np.dot(xhat, yr) / np.dot(xhat, xhat))

    rng = np.random.default_rng(0)
    bs = []
    for _ in range(60):
        i = rng.choice(len(d), len(d) // 4, replace=False)
        zi, xi, yi = resid(z[i], C[i]), resid(x[i], C[i]), resid(y[i], C[i])
        f = np.dot(zi, xi) / np.dot(zi, zi)
        xh = zi * f
        bs.append(np.dot(xh, yi) / np.dot(xh, xh) * 60 * 100)
    lo, hi = np.percentile(bs, [2.5, 97.5])

    print(f"\n  --- effect of +1 min headway on P(gain > {thr} s) ---")
    print(f"  n = {len(d):,}   controls: hour, month, track, direction, service")
    print(f"  OLS   {ols * 60 * 100:+.2f} percentage points per extra minute")
    print(f"  2SLS  {tsls * 60 * 100:+.2f} pp per extra minute   "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]")


def q3_dose_response(D: pd.DataFrame) -> None:
    print("\n  --- dose-response: P(moved) as trains separate ---")
    print(f"  {'headway':<12} {'n':>8} {'P(gain>60s)':>12} {'P(gain>120s)':>13}")
    bands = [(0, 120, "<2 min"), (120, 180, "2-3"), (180, 240, "3-4"),
             (240, 300, "4-5"), (300, 420, "5-7"), (420, 600, "7-10"),
             (600, 900, "10-15"), (900, 1200, "15-20"), (1200, 2400, "20-40"),
             (2400, 7200, "40+")]
    for lo, hi, lab in bands:
        b = D[D["act_hw"].between(lo, hi, inclusive="left")]
        if len(b) < 300:
            continue
        print(f"  {lab:<12} {len(b):>8,} {(b['foll_gain'] > 60).mean() * 100:>11.1f}% "
              f"{(b['foll_gain'] > 120).mean() * 100:>12.1f}%")


def main() -> None:
    D = build()
    D = D.dropna(subset=["foll_gain", "act_hw", "leader_entry", "foll_entry"])
    print(f"traversals: {len(D):,}")
    print("=" * 78)
    print("Q1 - can we identify the trains the junction actually moves?")
    print("=" * 78)
    for thr in THRESHOLDS:
        q1_predictive(D, thr)

    print("\n" + "=" * 78)
    print("Q2 - causal effect of headway on the probability of being moved")
    print("=" * 78)
    for thr in THRESHOLDS:
        q2_causal(D, thr)

    print("\n" + "=" * 78)
    print("Q3 - dose-response")
    print("=" * 78)
    q3_dose_response(D)


if __name__ == "__main__":
    main()
