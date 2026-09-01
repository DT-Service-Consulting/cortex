"""Where inside the junction is the delay actually lost - and do chains start at Central?

delay_gained = p5_delay_arr - p1_delay_dep decomposes exactly into 7 additive
terms (4 running sections, 3 intermediate dwells), so attribution needs no model:

    run p1->p2 | dwell p2 | run p2->p3 | dwell p3 | run p3->p4 | dwell p4 | run p4->p5

Central (215) is always p3 in both directions (see build_traversals.ORDER), so
'lost at Gare Centrale' is the dwell-p3 term and 'lost approaching Central' is
run p2->p3. The sum is checked against delay_gained as an identity.

Usage: python analysis/week2/where_lost.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "junction" / "traversals_202502.csv"
HW_MAX, THR = 300, 60

LEGS = ["run p1->p2", "dwell p2", "run p2->p3", "dwell p3 (CENTRAL)",
        "run p3->p4", "dwell p4", "run p4->p5"]


def secs(t):
    if not isinstance(t, str) or not t.strip():
        return float("nan")
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def load():
    df = pd.read_csv(P, dtype=str)
    df["entry_s"] = df["p1_real_dep"].map(secs)
    for c in ("entry_delay", "exit_delay", "delay_gained"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for p in range(1, 6):
        for f in ("delay_arr", "delay_dep"):
            df[f"p{p}_{f}"] = pd.to_numeric(df[f"p{p}_{f}"], errors="coerce")
    df["service"] = df["relation"].str.strip().str.split().str[0]

    df["run p1->p2"] = df["p2_delay_arr"] - df["p1_delay_dep"]
    df["dwell p2"] = df["p2_delay_dep"] - df["p2_delay_arr"]
    df["run p2->p3"] = df["p3_delay_arr"] - df["p2_delay_dep"]
    df["dwell p3 (CENTRAL)"] = df["p3_delay_dep"] - df["p3_delay_arr"]
    df["run p3->p4"] = df["p4_delay_arr"] - df["p3_delay_dep"]
    df["dwell p4"] = df["p4_delay_dep"] - df["p4_delay_arr"]
    df["run p4->p5"] = df["p5_delay_arr"] - df["p4_delay_dep"]
    return df


def chains(df):
    df = df.dropna(subset=["entry_s", "entry_delay", "exit_delay", "delay_gained"])
    df = df.sort_values(["date", "direction", "tunnel_track", "entry_s"])
    df = df.assign(moved=df["delay_gained"] > THR)
    out = []
    for _, g in df.groupby(["date", "direction", "tunnel_track"], sort=False):
        g = g.reset_index(drop=True)
        hw = g["entry_s"].diff()
        linked = (hw <= HW_MAX) & g["moved"] & g["moved"].shift(1).fillna(False)
        cid = (~linked).cumsum()
        for _, run in g.groupby(cid):
            run = run[run["moved"]]
            if len(run) >= 2:
                out.append(run)
    return out


def profile(sub, label):
    ok = sub.dropna(subset=LEGS)
    tot = ok[LEGS].sum()
    print(f"\n{label}  (n = {len(ok):,} of {len(sub):,} with all 7 legs)")
    resid = (ok[LEGS].sum(axis=1) - ok["delay_gained"]).abs().max()
    print(f"  decomposition residual vs delay_gained: max {resid:.0f}s")
    grand = tot.sum()
    for leg in LEGS:
        share = tot[leg] / grand * 100 if grand else 0
        bar = "#" * int(max(share, 0) / 2)
        print(f"    {leg:<20} {tot[leg]/len(ok):>7.1f}s/train  {share:>5.1f}%  {bar}")
    dom = ok[LEGS].idxmax(axis=1).value_counts()
    print("  single largest contributor per train:")
    for leg in LEGS:
        n = int(dom.get(leg, 0))
        print(f"    {leg:<20} {n:>6}  ({n/len(ok)*100:>5.1f}%)")


def main():
    df = load()
    print(f"February 2025: {len(df):,} traversals")
    assert (df["p3_name"].dropna() == "central").all(), "p3 is not always Central"
    print("checked: p3 is Brussels-Central (215) in every row, both directions")

    profile(df, "ALL traversals")
    cs = chains(df)
    print(f"\n{'='*70}\nchains of 2+: {len(cs):,}")
    allc = pd.concat(cs)
    profile(allc, "ALL trains inside a chain")
    profile(pd.concat([c.iloc[[0]] for c in cs]), "CHAIN STARTERS only (first moved train)")
    profile(pd.concat([c.iloc[1:] for c in cs]), "CHAIN FOLLOWERS (position 2+)")

    # how often is Central the dominant loss point, by chain length
    print(f"\n{'='*70}\nshare of chain STARTERS whose biggest single loss is at Central, by chain length")
    for lo, hi in [(2, 2), (3, 4), (5, 7), (8, 99)]:
        sel = [c.iloc[[0]] for c in cs if lo <= len(c) <= hi]
        if not sel:
            continue
        s = pd.concat(sel).dropna(subset=LEGS)
        dom = s[LEGS].idxmax(axis=1)
        n_c = (dom == "dwell p3 (CENTRAL)").sum()
        n_a = (dom == "run p2->p3").sum()
        lab = f"len {lo}" if lo == hi else f"len {lo}-{hi if hi < 99 else '+'}"
        print(f"  {lab:<10} n={len(s):>5}   at Central {n_c/len(s)*100:>5.1f}%   "
              f"approaching Central {n_a/len(s)*100:>5.1f}%")


if __name__ == "__main__":
    main()
