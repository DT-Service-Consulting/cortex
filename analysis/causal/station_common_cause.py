"""Is 'station' (specifically Central, the identified bottleneck) a common cause
behind the same-track chain/knock-on findings, rather than train-to-train relay?

Two trains on DIFFERENT tracks cannot physically queue behind each other - there
is no shared block section, no shared platform occupancy in the queueing sense.
So if delay at Central correlates across tracks for trains passing through in the
same real-world time window, that correlation cannot be relay/knock-on. It has to
be something hitting the station itself (a platform problem, a signal fault, a
dispatcher decision) that catches whoever is there at the time.

Method: for every train's arrival delay AT CENTRAL (the identified bottleneck),
find its nearest neighbor in real time on (a) the SAME track, (b) a DIFFERENT
track, both restricted to the same date and a <=10 min window. Correlate. Compare
against the same-date placebo baseline (different date, matched hour) already
established at r=0.024.

Usage: python station_common_cause.py
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
from run_causal import normal_scores  # noqa: E402

WINDOW = 600  # 10 min


def build() -> pd.DataFrame:
    df = load_all()
    df["p3_arr_s"] = df["p3_real_arr"].map(secs)
    df["p3_delay_arr"] = pd.to_numeric(df["p3_delay_arr"], errors="coerce")
    df = df.dropna(subset=["p3_arr_s", "p3_delay_arr"])
    return df


def nearest_neighbor(day: pd.DataFrame, same_track: bool) -> np.ndarray:
    """For each row, the p3_delay_arr of its nearest OTHER train in time,
    restricted to same_track=True/False and within WINDOW seconds."""
    day = day.sort_values("p3_arr_s").reset_index(drop=True)
    t = day["p3_arr_s"].to_numpy()
    trk = day["tunnel_track"].to_numpy()
    delay = day["p3_delay_arr"].to_numpy()
    out = np.full(len(day), np.nan)

    for i in range(len(day)):
        lo, hi = i - 1, i + 1
        best_j, best_dt = -1, WINDOW + 1
        # scan outward a bounded number of steps (data is dense; 40 is ample)
        for j in list(range(max(0, i - 40), i)) + list(range(i + 1, min(len(day), i + 41))):
            if (trk[j] == trk[i]) != same_track:
                continue
            dt = abs(t[j] - t[i])
            if dt <= WINDOW and dt < best_dt:
                best_dt, best_j = dt, j
        if best_j >= 0:
            out[i] = delay[best_j]
    return out


def main() -> None:
    df = build()
    print(f"trains with a valid Central arrival delay: {len(df):,}")

    same_all, cross_all, self_all = [], [], []
    for date, day in df.groupby("date"):
        if len(day) < 5:
            continue
        s = nearest_neighbor(day, same_track=True)
        c = nearest_neighbor(day, same_track=False)
        same_all.append(pd.Series(s, index=day.index))
        cross_all.append(pd.Series(c, index=day.index))
        self_all.append(day["p3_delay_arr"])

    self_d = pd.concat(self_all)
    same_d = pd.concat(same_all)
    cross_d = pd.concat(cross_all)

    print("\n=== correlation with THIS train's Central arrival delay ===")
    for label, other in (("same track, nearest in time (<=10min)", same_d),
                         ("DIFFERENT track, nearest in time (<=10min)", cross_d)):
        m = self_d.notna() & other.notna()
        x = normal_scores(self_d[m].to_numpy(float))
        y = normal_scores(other[m].to_numpy(float))
        r = np.corrcoef(x, y)[0, 1]
        print(f"  {label:<44} n={m.sum():>7,}  r={r:+.4f}")

    print("\n  placebo (different date, matched hour, from earlier test): r=+0.024")
    print("\nIf the cross-track number sits near the placebo -> the effect is track-")
    print("specific (relay/queueing). If it sits well above placebo -> station-level")
    print("congestion is implicated as a shared cause, independent of any specific")
    print("train-to-train relationship.")


if __name__ == "__main__":
    main()
