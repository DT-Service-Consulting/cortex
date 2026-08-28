"""Do delay cascades ('dominoes') actually exist in this data?

A chain: consecutive trains on the same track, each 'moved' (gained > 60s), each
following the previous with headway <= 5 min (the range where knock-on is strong).
If most moved events are isolated (chain length 1), the effect dissipates after one
hop and 'domino' framing is not supported. If chains of 2+ are common, cascades are
real and predicting them is a legitimate task.
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

HW_MAX = 300  # 5 min


def main() -> None:
    df = load_all()
    df["entry_s"] = df["p1_real_dep"].map(secs)
    df["exit_delay"] = pd.to_numeric(df["exit_delay"], errors="coerce")
    df["entry_delay"] = pd.to_numeric(df["entry_delay"], errors="coerce")
    df["delay_gained"] = pd.to_numeric(df["delay_gained"], errors="coerce")
    df = df.dropna(subset=["entry_s", "exit_delay", "entry_delay", "delay_gained"])
    df = df.sort_values(["date", "direction", "tunnel_track", "entry_s"])
    df["moved"] = df["delay_gained"] > 60

    runs = []
    for _, g in df.groupby(["date", "direction", "tunnel_track"], sort=False):
        g = g.reset_index(drop=True)
        hw = g["entry_s"].diff()
        linked = (hw <= HW_MAX) & g["moved"] & g["moved"].shift(1).fillna(False)
        chain_id = (~linked).cumsum()
        for _, run in g[g["moved"]].groupby(chain_id[g["moved"]]):
            runs.append(len(run))

    runs = np.array(runs)
    print(f"total 'moved' events            : {df['moved'].sum():,}")
    print(f"events grouped into chains       : {len(runs):,} chains")
    print("\nchain length distribution (headway <= 5 min between links):")
    for L in range(1, 7):
        n = (runs == L).sum()
        print(f"  length {L}: {n:>7,} chains  ({n / len(runs) * 100:5.1f}%)   "
              f"covering {n * L:,} trains")
    print(f"  length >=7: {(runs >= 7).sum():,} chains, max length {runs.max()}")

    isolated = (runs == 1).sum()
    chained = (runs >= 2).sum()
    trains_in_2plus = runs[runs >= 2].sum()
    print(f"\nisolated moved events (no domino): {isolated:,} ({isolated / len(runs) * 100:.1f}% of chains)")
    print(f"chains of 2+                     : {chained:,} ({chained / len(runs) * 100:.1f}% of chains)")
    print(f"trains caught in a 2+ chain       : {trains_in_2plus:,} "
          f"({trains_in_2plus / df['moved'].sum() * 100:.1f}% of all moved trains)")

    # unconditional persistence: given train i moved, P(i+1 also moved)?
    print("\n=== unconditional persistence ===")
    p_move = df["moved"].mean()
    print(f"baseline P(any train moved)         = {p_move * 100:.1f}%")
    nxt_moved, nxt_hw, cur_moved = [], [], []
    for _, g in df.groupby(["date", "direction", "tunnel_track"], sort=False):
        g = g.reset_index(drop=True)
        cur_moved.extend(g["moved"].iloc[:-1].tolist())
        nxt_moved.extend(g["moved"].iloc[1:].tolist())
        nxt_hw.extend(g["entry_s"].diff().iloc[1:].tolist())
    t = pd.DataFrame({"cur": cur_moved, "nxt": nxt_moved, "hw": nxt_hw}).dropna()
    for lo, hi, lab in [(0, 180, "<3min"), (180, 300, "3-5"), (300, 600, "5-10"), (600, 99999, ">10")]:
        b = t[t["hw"].between(lo, hi)]
        pm = b[b["cur"]]["nxt"].mean()
        pu = b[~b["cur"]]["nxt"].mean()
        print(f"  headway {lab:<7}  P(next moved | this moved)={pm * 100:5.1f}%   "
              f"P(next moved | this NOT moved)={pu * 100:5.1f}%   ratio={pm / pu:.2f}x")


if __name__ == "__main__":
    main()
