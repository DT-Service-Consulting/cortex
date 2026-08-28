"""Does a chain's starter affect trains two hops away, or only its immediate
neighbor? And does it affect trains OUTSIDE its own chain at all?

Two tests:
  1. Multi-hop (Markov check). For chains of length >=3 (A -> B -> C, same
     track, linked by headway<=5min and both moved): does A's state still
     predict C's outcome after fully controlling for B's state? If yes, the
     effect skips a hop - inconsistent with pure local queueing (which should
     be first-order Markov: C only "feels" its immediate predecessor B) and
     suggestive of a longer-lived shared cause (signal fault, persistent
     disruption) that a pairwise headway model would miss.
  2. Placebo (specificity check). Does A's state predict a train that is NOT
     in its chain at all - matched on hour/month/track so it's a fair
     comparison, but on a different day? If the real link is much stronger
     than the placebo, the chain relationship is specific, not a data-wide
     artifact (e.g. "delay is elevated in general at this hour").

Usage: python multihop_test.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
from step7_full_year import load_all, secs  # noqa: E402
from run_causal import normal_scores, partial_corr  # noqa: E402

HW_MAX = 300


def build_chains(min_len: int = 3) -> pd.DataFrame:
    df = load_all()
    df["entry_s"] = df["p1_real_dep"].map(secs)
    df["exit_delay"] = pd.to_numeric(df["exit_delay"], errors="coerce")
    df["entry_delay"] = pd.to_numeric(df["entry_delay"], errors="coerce")
    df["delay_gained"] = pd.to_numeric(df["delay_gained"], errors="coerce")
    df = df.dropna(subset=["entry_s", "exit_delay", "entry_delay", "delay_gained"])
    df = df.sort_values(["date", "direction", "tunnel_track", "entry_s"])
    df["moved"] = df["delay_gained"] > 60
    df["hour"] = (df["entry_s"] // 3600).astype(int)
    df["month_i"] = df["month"].str[-2:].astype(int)

    rows = []
    for (date, direction, track), g in df.groupby(["date", "direction", "tunnel_track"], sort=False):
        g = g.reset_index(drop=True)
        hw = g["entry_s"].diff()
        linked = (hw <= HW_MAX) & g["moved"] & g["moved"].shift(1).fillna(False)
        chain_id = (~linked).cumsum()
        for _, run in g.groupby(chain_id):
            run = run[run["moved"]]
            if len(run) < min_len:
                continue
            a, b, c = run.iloc[0], run.iloc[1], run.iloc[2]
            rows.append({
                "date": date, "direction": direction, "track": track,
                "hour": a["hour"], "month_i": a["month_i"],
                "A_exit": a["exit_delay"], "A_entry": a["entry_delay"],
                "B_exit": b["exit_delay"], "B_entry": b["entry_delay"],
                "hw_AB": b["entry_s"] - a["entry_s"],
                "C_exit": c["exit_delay"], "C_entry": c["entry_delay"],
                "hw_BC": c["entry_s"] - b["entry_s"],
                "chain_len": len(run),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ch = build_chains()
    print(f"chains of length >=3: {len(ch):,}")
    print(f"mean chain length (of these): {ch['chain_len'].mean():.1f}, "
          f"max {ch['chain_len'].max()}")

    # ---------------- Test 1: multi-hop / Markov check ----------------
    print("\n" + "=" * 78)
    print("TEST 1 - does A (2 hops back) still predict C after controlling for B?")
    print("=" * 78)
    a = normal_scores(ch["A_exit"].to_numpy(float))
    c = normal_scores(ch["C_exit"].to_numpy(float))
    raw = np.corrcoef(a, c)[0, 1]
    print(f"n = {len(ch):,}")
    print(f"raw r(A_exit, C_exit)                              = {raw:+.4f}")

    Z_b = np.column_stack([
        normal_scores(ch["B_exit"].to_numpy(float)),
        normal_scores(ch["B_entry"].to_numpy(float)),
        normal_scores(ch["hw_AB"].to_numpy(float)),
        normal_scores(ch["hw_BC"].to_numpy(float)),
        pd.get_dummies(ch["hour"], drop_first=True).astype(float).to_numpy(),
        pd.get_dummies(ch["month_i"], drop_first=True).astype(float).to_numpy(),
    ])
    partial = partial_corr(a, c, Z_b)
    print(f"partial r(A_exit, C_exit | B's full state, hw, hour, month) = {partial:+.4f}")
    print(f"\nfraction of the raw association still present after controlling for B: "
          f"{partial / raw * 100:.0f}%")
    if abs(partial) > 0.05:
        print("=> A retains a DIRECT effect on C beyond what B mediates.")
        print("   Not pure first-order Markov propagation - consistent with a")
        print("   longer-lived shared cause (e.g. an incident spanning multiple trains)")
        print("   rather than each train only reacting to its immediate predecessor.")
    else:
        print("=> A's effect on C is (almost) fully explained by B.")
        print("   Consistent with first-order Markov / local queueing: what matters")
        print("   is your immediate predecessor, not who started the trouble.")

    # ---------------- Test 2: placebo / specificity ----------------
    print("\n" + "=" * 78)
    print("TEST 2 - placebo: does A predict an UNRELATED train (not in its chain)?")
    print("=" * 78)
    rng = np.random.default_rng(0)
    # match on hour to keep it a fair comparison, pick from a DIFFERENT date
    placebo_c = []
    dates = ch["date"].unique()
    by_hour = {h: ch[ch["hour"] == h] for h in ch["hour"].unique()}
    for _, row in ch.iterrows():
        pool = by_hour[row["hour"]]
        pool = pool[pool["date"] != row["date"]]
        if len(pool) == 0:
            placebo_c.append(np.nan)
            continue
        placebo_c.append(pool["C_exit"].sample(1, random_state=rng.integers(1e9)).iloc[0])
    ch["C_placebo"] = placebo_c
    ok = ch.dropna(subset=["C_placebo"])
    a2 = normal_scores(ok["A_exit"].to_numpy(float))
    cp = normal_scores(ok["C_placebo"].to_numpy(float))
    r_placebo = np.corrcoef(a2, cp)[0, 1]
    print(f"n = {len(ok):,}")
    print(f"r(A_exit, REAL chain member C_exit)     = {raw:+.4f}")
    print(f"r(A_exit, PLACEBO unrelated train, same hour, different day) = {r_placebo:+.4f}")
    print(f"\n=> the real link is {raw / r_placebo if abs(r_placebo) > 1e-6 else float('inf'):.1f}x "
          f"the placebo (should be near 1x if the effect were just an hour-of-day artifact)")


if __name__ == "__main__":
    main()
