"""Step 1 - leader/follower pair table on shared tunnel track.

A 'pair' is two consecutive traversals on the same (date, direction,
tunnel_track). Only features knowable at the follower's entry are carried
forward as leader_* - the leader is usually still inside the junction when the
follower enters (median headway 5.3 min < median traversal 8.8 min), so the
leader's exit is NOT available at prediction time.
"""
from __future__ import annotations

import pandas as pd

from common import OUT, load_traversals

MAX_HEADWAY = 7200  # ignore gaps beyond 2h (overnight breaks)


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "direction", "tunnel_track", "entry_s"])
    g = df.groupby(["date", "direction", "tunnel_track"], sort=False)

    out = pd.DataFrame({
        "date": df["date"],
        "direction": df["direction"],
        "tunnel_track": df["tunnel_track"],
        "hour": df["hour"],
        "service": df["service"],
        "follower_train": df["train_no"],
        "follower_entry_s": df["entry_s"],
        "follower_entry_delay": df["entry_delay"],
        "follower_exit_delay": df["exit_delay"],
        "follower_delay_gained": df["delay_gained"],
        "leader_train": g["train_no"].shift(1),
        "leader_entry_s": g["entry_s"].shift(1),
        "leader_entry_delay": g["entry_delay"].shift(1),
        # kept for analysis only - NOT a valid predictor, see module docstring
        "leader_exit_delay_POSTHOC": g["exit_delay"].shift(1),
    })
    out["headway_s"] = out["follower_entry_s"] - out["leader_entry_s"]
    out = out[out["headway_s"].between(1, MAX_HEADWAY)]
    return out.reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_traversals()
    pairs = build(df)
    path = OUT / "pairs_202501.csv"
    pairs.to_csv(path, index=False)

    print(f"traversals in    : {len(df):,}")
    print(f"pairs out        : {len(pairs):,}")
    print(f"-> {path}")

    h = pairs["headway_s"]
    print(f"\nheadway (s): p10={h.quantile(.1):.0f} p50={h.quantile(.5):.0f} "
          f"p90={h.quantile(.9):.0f}  (median {h.median()/60:.1f} min)")

    print("\nknock-on signal by headway band")
    print("  correlations use leader_exit_delay_POSTHOC - descriptive only, not a predictor")
    bands = [(0, 300, "<5min"), (300, 600, "5-10min"),
             (600, 1200, "10-20min"), (1200, 3600, "20-60min")]
    print(f"  {'band':<10} {'n':>7} {'r(lead_exit, foll_gain)':>24} {'r(lead_entry, foll_gain)':>26}")
    for lo, hi, lab in bands:
        b = pairs[pairs["headway_s"].between(lo, hi, inclusive="left")]
        r1 = b["leader_exit_delay_POSTHOC"].corr(b["follower_delay_gained"])
        r2 = b["leader_entry_delay"].corr(b["follower_delay_gained"])
        print(f"  {lab:<10} {len(b):>7,} {r1:>24.3f} {r2:>26.3f}")


if __name__ == "__main__":
    main()
