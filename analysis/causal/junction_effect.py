"""Does the junction change the delay at all?

Everything so far assumed it does. But the median train gains ~7 s crossing a
3.5 km tunnel, which is close to nothing. Before asking *why* delay propagates,
test whether the junction moves the delay distribution at all - and if it only
does so for a minority, isolate that minority and characterise it.

Four questions:
  1. Are the entry and exit delay distributions different? (effect size, not p)
  2. What share of trains is essentially unaffected?
  3. How much of the apparent "effect" is mean reversion, i.e. late trains
     recovering into schedule padding and early trains giving time back?
  4. For the trains that ARE moved, what distinguishes them?

Usage: python junction_effect.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from run_causal import build

QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def q1_distributions(D: pd.DataFrame) -> None:
    a = D["foll_entry"].to_numpy(float)   # delay entering the junction
    b = D["exit_delay"].to_numpy(float)   # delay leaving it
    print("=" * 78)
    print("Q1 - are the entry and exit delay distributions different?")
    print("=" * 78)
    print(f"  n = {len(a):,}\n")
    print(f"  {'quantile':>9} {'entry (s)':>11} {'exit (s)':>10} {'shift':>8}")
    for q in QS:
        qa, qb = np.quantile(a, q), np.quantile(b, q)
        print(f"  {q:>9.2f} {qa:>11.0f} {qb:>10.0f} {qb - qa:>+8.0f}")
    print(f"\n  {'mean':>9} {a.mean():>11.0f} {b.mean():>10.0f} {b.mean() - a.mean():>+8.0f}")
    print(f"  {'sd':>9} {a.std():>11.0f} {b.std():>10.0f} {b.std() - a.std():>+8.0f}")

    # Effect size, not significance: at n=313k every test rejects.
    d = (b - a)
    cohen = d.mean() / d.std()
    ks = stats.ks_2samp(a[:50000], b[:50000]).statistic
    print(f"\n  paired mean shift  : {d.mean():+.1f} s   (sd of shift {d.std():.0f} s)")
    print(f"  standardised effect: {cohen:+.3f}   <- Cohen's d on the paired difference")
    print(f"  KS statistic       : {ks:.4f}   (on 50k subsample)")
    print("  Note: a paired t-test here returns p ~ 0 for a shift of a few seconds.")
    print("        n = 313k makes significance meaningless; read the effect sizes.")


def q2_unaffected(D: pd.DataFrame) -> None:
    g = D["foll_gain"].to_numpy(float)
    print("\n" + "=" * 78)
    print("Q2 - what share of trains is essentially unaffected?")
    print("=" * 78)
    for thr in (15, 30, 60, 120):
        print(f"  |delay gained| <= {thr:>3} s : {(np.abs(g) <= thr).mean() * 100:5.1f}%")
    print(f"\n  gained more than  +60 s : {(g > 60).mean() * 100:5.1f}%")
    print(f"  lost   more than  -60 s : {(g < -60).mean() * 100:5.1f}%")
    share = g[g > 60].sum() / g.sum() * 100
    print(f"\n  the {(g > 60).mean() * 100:.1f}% gaining >60 s account for "
          f"{share:.0f}% of all delay added by the junction")


def q3_mean_reversion(D: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("Q3 - is the 'effect' just mean reversion into schedule padding?")
    print("=" * 78)
    e = D["foll_entry"].to_numpy(float)
    g = D["foll_gain"].to_numpy(float)
    print(f"  corr(entry delay, delay gained) = {np.corrcoef(e, g)[0, 1]:+.3f}")
    print("  A negative value means late trains shed delay and early trains give it back,")
    print("  which is what recovery margin in the timetable produces mechanically.\n")
    dec = pd.qcut(D["foll_entry"], 10, labels=False, duplicates="drop")
    t = D.assign(dec=dec).groupby("dec").agg(
        entry=("foll_entry", "mean"), gain=("foll_gain", "mean"),
        gain_med=("foll_gain", "median"), n=("foll_gain", "size"))
    print(f"  {'decile':>7} {'mean entry':>11} {'mean gain':>10} {'median gain':>12} {'n':>8}")
    for i, r in t.iterrows():
        print(f"  {int(i) + 1:>7} {r['entry']:>11.0f} {r['gain']:>10.0f} "
              f"{r['gain_med']:>12.0f} {int(r['n']):>8,}")


def q4_who_is_moved(D: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("Q4 - for the trains that ARE moved, what distinguishes them?")
    print("=" * 78)
    D = D.copy()
    D["moved"] = D["foll_gain"] > 60
    print(f"  'moved' = gained more than 60 s   ({D['moved'].mean() * 100:.1f}% of traversals)\n")

    num = {"act_hw": "headway (s)", "foll_entry": "own entry delay (s)",
           "leader_entry": "leader entry delay (s)", "hour": "hour of day",
           "temp": "temperature (C)"}
    print(f"  {'variable':<26} {'unmoved':>10} {'moved':>10} {'diff':>10}")
    for c, lab in num.items():
        u, m = D.loc[~D["moved"], c].mean(), D.loc[D["moved"], c].mean()
        print(f"  {lab:<26} {u:>10.1f} {m:>10.1f} {m - u:>+10.1f}")

    print("\n  share moved, by headway band:")
    for lo, hi, lab in [(0, 180, "<3 min"), (180, 300, "3-5"), (300, 600, "5-10"),
                        (600, 1200, "10-20"), (1200, 7200, ">20")]:
        b = D[D["act_hw"].between(lo, hi, inclusive="left")]
        if len(b) > 300:
            print(f"    {lab:<9} n={len(b):>7,}  {b['moved'].mean() * 100:5.1f}%")

    print("\n  share moved, by tunnel track:")
    for t, b in D.groupby("tunnel_track"):
        print(f"    {t:<9} n={len(b):>7,}  {b['moved'].mean() * 100:5.1f}%")

    print("\n  share moved, by service:")
    top = D["service"].value_counts().head(6).index
    for s in top:
        b = D[D["service"] == s]
        print(f"    {s:<9} n={len(b):>7,}  {b['moved'].mean() * 100:5.1f}%")


def q5_causal_on_moved(D: pd.DataFrame) -> None:
    """Does headway matter more inside the affected subpopulation?"""
    from run_causal import normal_scores, partial_corr
    print("\n" + "=" * 78)
    print("Q5 - is the headway effect concentrated in the moved subpopulation?")
    print("=" * 78)
    ctrl = ["hour", "month_i", "temp", "rain", "snow"]
    for lab, sub in (("all traversals", D),
                     ("moved only (gain > 60 s)", D[D["foll_gain"] > 60]),
                     ("unmoved (|gain| <= 60 s)", D[D["foll_gain"].abs() <= 60])):
        s = sub[sub["act_hw"] < 300]
        if len(s) < 500:
            continue
        x = normal_scores(s["leader_entry"].to_numpy(float))
        y = normal_scores(s["foll_gain"].to_numpy(float))
        z = normal_scores(s["act_hw"].to_numpy(float))
        Z = s[ctrl].to_numpy(float)
        print(f"  {lab:<26} n={len(s):>7,}  "
              f"r(leader,gain)={partial_corr(x, y, Z):+.3f}  "
              f"r(headway,gain)={partial_corr(z, y, Z):+.3f}")


def main() -> None:
    D = build()
    D["exit_delay"] = pd.to_numeric(D["exit_delay"], errors="coerce")
    D = D.dropna(subset=["foll_entry", "exit_delay", "foll_gain"])
    q1_distributions(D)
    q2_unaffected(D)
    q3_mean_reversion(D)
    q4_who_is_moved(D)
    q5_causal_on_moved(D)


if __name__ == "__main__":
    main()
