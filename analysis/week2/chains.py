"""Canonical definition of a delay chain, and extraction of the February top 10.

CHAIN DEFINITION (identical to analysis/causal/cascade_check.py, restated here
so this folder is self-contained and the parameters are explicit):

  scope   trains sharing (date, direction, tunnel_track) - the same physical
          tunnel track, same direction, same day - sorted by entry time.
          tunnel_track is LINE_NO_DEP read at Brussels-Central; see data/README.md.
  moved   delay_gained > THR, where delay_gained = p5_delay_arr - p1_delay_dep.
  link    train i links to train i-1 iff BOTH are moved AND the entry headway
          between them is <= HW_MAX.
  chain   walk the sorted sequence; every broken link starts a new chain.
          A chain is reported here only if it has >= 2 members.

  Known brittleness: the link uses a hard threshold on both ends, so a cascade
  passing through a train that lands just under THR is split into two chains.
  Chain lengths are therefore a LOWER BOUND. This is why the extracts carry the
  sub-threshold trains that bounded each chain (role 'breaker_*') - they are the
  rows that answer "why did it start here" and "why did it stop here".

  entry_s is seconds since midnight WITHIN a DATDEP group; a traversal starting
  before midnight and ending after keeps its DATDEP, so entry_s never wraps
  inside a group. Irrelevant for these 08:xx/17:xx chains, but do not reuse
  entry_s across dates.

SELECTION: the 10 chains with the greatest length, ties broken by total
delay_gained. Rank is positional and unstable across parameter changes - join on
chain_id, which is content-derived.

DELAY DECOMPOSITION: delay_gained decomposes exactly (residual 0s) into
  run p1->p2 | dwell p2 | run p2->p3 | dwell p3 | run p3->p4 | dwell p4 | run p4->p5
Central (215) is always p3 in both directions (build_traversals.ORDER). A dwell
term is delay_dep - delay_arr = actual_dwell - planned_dwell, i.e. it is EXCESS
dwell; planned and actual dwell are carried separately so it can be read.

Outputs:
  analysis/week2/chains_top10.csv          manifest, 10 rows (committable)
  data/week2/chain_members.csv             one row per (chain, position, train)
  data/week2/timetable/<chain_id>.csv      long-form 5-point timetable per chain,
                                           members + breakers + context
  data/week2/control_days.csv              same scheduled trains on every other
                                           February day - the counterfactual

Usage: python analysis/week2/chains.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WIDE = ROOT / "data" / "junction" / "traversals_202502.csv"
LONG = ROOT / "data" / "junction" / "junction_traversing_202502.csv"
OUT_DATA = ROOT / "data" / "week2"
OUT_HERE = Path(__file__).resolve().parent

THR = 60           # seconds of delay_gained above which a train is 'moved'
HW_MAX = 300       # seconds of entry headway for two moved trains to link
GROUP = ["date", "direction", "tunnel_track"]
TOP_N = 10
CONTEXT_S = 1800   # +/- 30 min of same-group traversals carried as context

LEGS = ["run_p1_p2", "dwell_p2", "run_p2_p3", "dwell_p3_central",
        "run_p3_p4", "dwell_p4", "run_p4_p5"]

MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def secs(t) -> float:
    """Seconds since midnight. Hours are not zero-padded in this feed."""
    if not isinstance(t, str) or not t.strip():
        return float("nan")
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def hhmmss(s) -> str:
    s = int(s) % 86400
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def iso_date(d: str) -> str:
    """'07FEB2025' -> '2025-02-07'."""
    return f"{d[5:]}-{MONTHS[d[2:5]]:02d}-{int(d[:2]):02d}"


def load_wide() -> pd.DataFrame:
    df = pd.read_csv(WIDE, dtype=str)
    df["entry_s"] = df["p1_real_dep"].map(secs)
    for c in ("entry_delay", "exit_delay", "delay_gained"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for p in range(1, 6):
        for f in ("delay_arr", "delay_dep"):
            df[f"p{p}_{f}"] = pd.to_numeric(df[f"p{p}_{f}"], errors="coerce")
    df["service"] = df["relation"].str.strip().str.split().str[0]
    df["iso_date"] = df["date"].map(iso_date)

    df["run_p1_p2"] = df["p2_delay_arr"] - df["p1_delay_dep"]
    df["dwell_p2"] = df["p2_delay_dep"] - df["p2_delay_arr"]
    df["run_p2_p3"] = df["p3_delay_arr"] - df["p2_delay_dep"]
    df["dwell_p3_central"] = df["p3_delay_dep"] - df["p3_delay_arr"]
    df["run_p3_p4"] = df["p4_delay_arr"] - df["p3_delay_dep"]
    df["dwell_p4"] = df["p4_delay_dep"] - df["p4_delay_arr"]
    df["run_p4_p5"] = df["p5_delay_arr"] - df["p4_delay_dep"]

    # a dwell term is EXCESS dwell; carry planned and actual so it can be read
    df["p3_planned_dwell_s"] = df["p3_planned_dep"].map(secs) - df["p3_planned_arr"].map(secs)
    df["p3_actual_dwell_s"] = df["p3_real_dep"].map(secs) - df["p3_real_arr"].map(secs)
    return df


def find_chains(df: pd.DataFrame, min_len: int = 2) -> list[pd.DataFrame]:
    d = df.dropna(subset=["entry_s", "entry_delay", "exit_delay", "delay_gained"])
    d = d.sort_values(GROUP + ["entry_s"])
    d = d.assign(moved=d["delay_gained"] > THR)
    out = []
    for _, g in d.groupby(GROUP, sort=False):
        g = g.reset_index(drop=True)
        hw = g["entry_s"].diff()
        linked = (hw <= HW_MAX) & g["moved"] & g["moved"].shift(1).fillna(False)
        cid = (~linked).cumsum()
        for _, run in g.groupby(cid):
            run = run[run["moved"]]
            if len(run) >= min_len:
                out.append(run.reset_index(drop=True))
    return out


def chain_id(c: pd.DataFrame) -> str:
    """Content-derived key: date, direction, track, start time. Stable across reruns."""
    f = c.iloc[0]
    track = str(f["tunnel_track"]).replace("/", "_")
    start = hhmmss(f["entry_s"])[:5].replace(":", "")
    return f"{iso_date(f['date']).replace('-', '')}-{f['direction']}-{track}-{start}"


def assign_clusters(top: list[pd.DataFrame], min_j: float = 0.5) -> dict[str, str]:
    """Group chains that are the same scheduled service on different days.

    Exact cast equality finds nothing - the cast varies day to day (a train
    cancelled, a different one routed onto the track). Single-linkage on Jaccard
    overlap of the member train_no sets does find them: the evening N2S 0/2 group
    and the morning S2N 0/3 group both sit at 0.64-0.83 pairwise.
    """
    ids = [chain_id(c) for c in top]
    casts = {i: set(c["train_no"]) for i, c in zip(ids, top)}
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(ids, 2):
        j = len(casts[a] & casts[b]) / len(casts[a] | casts[b])
        if j >= min_j:
            parent[find(a)] = find(b)

    roots = {}
    for i in ids:
        r = find(i)
        roots.setdefault(r, f"C{len(roots) + 1}")
    return {i: roots[find(i)] for i in ids}


def build_manifest(top: list[pd.DataFrame]) -> pd.DataFrame:
    clusters = assign_clusters(top)
    rows = []
    for rank, c in enumerate(top, 1):
        f, l = c.iloc[0], c.iloc[-1]
        hw = c["entry_s"].diff().dropna()
        members = c["train_no"].tolist()
        rows.append({
            "chain_id": chain_id(c),
            "rank": rank,
            "date": f["date"],
            "iso_date": f["iso_date"],
            "direction": f["direction"],
            "tunnel_track": f["tunnel_track"],
            "length": len(c),
            "start_time": hhmmss(f["entry_s"]),
            "end_time": hhmmss(l["entry_s"]),
            "span_s": int(l["entry_s"] - f["entry_s"]),
            "total_gained_s": int(c["delay_gained"].sum()),
            "max_gained_s": int(c["delay_gained"].max()),
            "mean_headway_s": round(hw.mean(), 1) if len(hw) else "",
            "min_headway_s": int(hw.min()) if len(hw) else "",
            "starter_train": f["train_no"],
            "terminator_train": l["train_no"],
            "members": "|".join(members),
            "service_mix": "|".join(f"{k}:{v}" for k, v in
                                    c["service"].value_counts().items()),
            # recurrence: exact cast key, plus a fuzzy cluster label (Jaccard >= 0.5)
            # for the same scheduled service running on a different day
            "recurrence_key": "|".join(sorted(members)),
            "cast_cluster": clusters[chain_id(c)],
            "thr_s": THR,
            "hw_max_s": HW_MAX,
            "group_key": "+".join(GROUP),
            "selection": f"top {TOP_N} by length, ties by total delay_gained",
        })
    return pd.DataFrame(rows)


def build_members(top: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for rank, c in enumerate(top, 1):
        cid = chain_id(c)
        hw = c["entry_s"].diff()
        for pos, (_, r) in enumerate(c.iterrows(), 1):
            rows.append({
                "chain_id": cid, "rank": rank, "position": pos,
                "date": r["date"], "iso_date": r["iso_date"],
                "direction": r["direction"], "tunnel_track": r["tunnel_track"],
                "train_no": r["train_no"], "relation": r["relation"],
                "service": r["service"],
                "entry_time": hhmmss(r["entry_s"]),
                "headway_prev_s": "" if pd.isna(hw.iloc[pos - 1]) else int(hw.iloc[pos - 1]),
                "entry_delay": int(r["entry_delay"]),
                "exit_delay": int(r["exit_delay"]),
                "delay_gained": int(r["delay_gained"]),
                **{k: ("" if pd.isna(r[k]) else int(r[k])) for k in LEGS},
                "p3_planned_dwell_s": "" if pd.isna(r["p3_planned_dwell_s"]) else int(r["p3_planned_dwell_s"]),
                "p3_actual_dwell_s": "" if pd.isna(r["p3_actual_dwell_s"]) else int(r["p3_actual_dwell_s"]),
                "p3_stop_type": r["p3_stop_type"],
                "entry_line": r["entry_line"], "exit_line": r["exit_line"],
            })
    return pd.DataFrame(rows)


def write_timetables(top: list[pd.DataFrame], wide: pd.DataFrame) -> None:
    """Per chain: the long-form 5-point rows for members, the two trains that
    bounded the chain, and the surrounding same-track traffic."""
    long = pd.read_csv(LONG, dtype=str)
    long["_key"] = long["DATDEP"] + "|" + long["TRAIN_NO"]

    tt_dir = OUT_DATA / "timetable"
    tt_dir.mkdir(parents=True, exist_ok=True)

    w = wide.dropna(subset=["entry_s"]).sort_values(GROUP + ["entry_s"])
    for c in top:
        cid = chain_id(c)
        f, l = c.iloc[0], c.iloc[-1]
        same = w[(w["date"] == f["date"]) & (w["direction"] == f["direction"]) &
                 (w["tunnel_track"] == f["tunnel_track"])].reset_index(drop=True)

        member_ids = set(c["train_no"])
        i_start = same.index[same["train_no"] == f["train_no"]][0]
        i_end = same.index[same["train_no"] == l["train_no"]][-1]

        roles = {}
        for i, r in same.iterrows():
            if r["train_no"] in member_ids and i_start <= i <= i_end:
                roles[r["train_no"]] = "member"
            elif i == i_start - 1:
                roles[r["train_no"]] = "breaker_before"
            elif i == i_end + 1:
                roles[r["train_no"]] = "breaker_after"
            elif f["entry_s"] - CONTEXT_S <= r["entry_s"] <= l["entry_s"] + CONTEXT_S:
                roles[r["train_no"]] = "context"

        sel = same[same["train_no"].isin(roles)].copy()
        # train_no keys the role/metadata lookup below; a repeat within one
        # (date, direction, track) group would silently corrupt it
        assert sel["train_no"].is_unique, f"{cid}: duplicate train_no in group"
        sel["role"] = sel["train_no"].map(roles)
        keys = set(sel["date"] + "|" + sel["train_no"])

        rows = long[long["_key"].isin(keys)].copy()
        meta = sel.set_index("train_no")
        rows["chain_id"] = cid
        rows["role"] = rows["TRAIN_NO"].map(meta["role"])
        rows["entry_delay"] = rows["TRAIN_NO"].map(meta["entry_delay"])
        rows["exit_delay"] = rows["TRAIN_NO"].map(meta["exit_delay"])
        rows["delay_gained"] = rows["TRAIN_NO"].map(meta["delay_gained"])
        rows["chain_position"] = rows["TRAIN_NO"].map(
            {t: i + 1 for i, t in enumerate(c["train_no"])})
        rows["entry_s"] = rows["TRAIN_NO"].map(meta["entry_s"])
        rows = rows.drop(columns=["_key"]).sort_values(["entry_s", "seq"])
        rows.to_csv(tt_dir / f"{cid}.csv", index=False)
        n_members = int((rows["role"] == "member").sum()) // 5
        print(f"  {cid}.csv  {len(rows):>4} rows  "
              f"({n_members} members, {rows['TRAIN_NO'].nunique()} trains)")


def build_control_days(top: list[pd.DataFrame], wide: pd.DataFrame,
                       all_chains: list[pd.DataFrame]) -> pd.DataFrame:
    """The counterfactual: the same scheduled cast, same track and direction, on
    every other February day. Same timetable, mostly no chain."""
    # (start_s, length) of every chain, so the day-wide and the time-windowed
    # maxima can be told apart: a 17:14 chain must not be credited with a
    # cascade that happened at 08:00 on the same track.
    by_group = {}
    for c in all_chains:
        f = c.iloc[0]
        key = (f["date"], f["direction"], f["tunnel_track"])
        by_group.setdefault(key, []).append((f["entry_s"], len(c)))

    rows = []
    for rank, c in enumerate(top, 1):
        cid = chain_id(c)
        f, l = c.iloc[0], c.iloc[-1]
        cast = set(c["train_no"])
        sub = wide[(wide["direction"] == f["direction"]) &
                   (wide["tunnel_track"] == f["tunnel_track"]) &
                   (wide["train_no"].isin(cast))]
        for date, g in sub.groupby("date"):
            found = by_group.get((date, f["direction"], f["tunnel_track"]), [])
            in_win = [n for s, n in found
                      if f["entry_s"] - CONTEXT_S <= s <= l["entry_s"] + CONTEXT_S]
            ran = len(g)
            moved = int((g["delay_gained"] > THR).sum())
            rows.append({
                "chain_id": cid, "rank": rank, "date": date,
                "iso_date": iso_date(date),
                "weekday": pd.Timestamp(iso_date(date)).day_name(),
                "is_weekday": pd.Timestamp(iso_date(date)).dayofweek < 5,
                "direction": f["direction"], "tunnel_track": f["tunnel_track"],
                "cast_size": len(cast),
                "cast_ran": ran,
                "cast_moved": moved,
                # the count is not comparable across days - the denominator
                # swings from 5 (weekend) to 12 (weekday); use the rate
                "cast_moved_rate": round(moved / ran, 3) if ran else "",
                "cast_total_gained_s": int(g["delay_gained"].sum()),
                "max_chain_len_in_window": max(in_win) if in_win else 0,
                "max_chain_len_that_day": max((n for _, n in found), default=0),
                "chain_day": date == f["date"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    wide = load_wide()
    print(f"February 2025: {len(wide):,} traversals, "
          f"{int((wide['delay_gained'] > THR).sum()):,} moved (>{THR}s)")

    resid = (wide[LEGS].sum(axis=1) - wide["delay_gained"]).abs().max()
    assert resid == 0, f"leg decomposition does not close: max residual {resid}s"
    assert (wide["p3_name"].dropna() == "central").all(), "p3 is not always Central"
    print(f"checked: 7-leg decomposition closes exactly (residual {resid:.0f}s); "
          f"p3 is Central in every row")

    all_chains = find_chains(wide)
    all_chains.sort(key=lambda c: (len(c), c["delay_gained"].sum()), reverse=True)
    top = all_chains[:TOP_N]
    print(f"chains of 2+: {len(all_chains):,}   -> top {TOP_N} "
          f"(lengths {[len(c) for c in top]})\n")

    man = build_manifest(top)
    man.to_csv(OUT_HERE / "chains_top10.csv", index=False)
    print(f"wrote {OUT_HERE / 'chains_top10.csv'}  ({len(man)} chains)")

    mem = build_members(top)
    mem.to_csv(OUT_DATA / "chain_members.csv", index=False)
    print(f"wrote {OUT_DATA / 'chain_members.csv'}  ({len(mem)} trains)\n")

    print("per-chain timetable extracts:")
    write_timetables(top, wide)

    ctl = build_control_days(top, wide, all_chains)
    ctl.to_csv(OUT_DATA / "control_days.csv", index=False)
    print(f"\nwrote {OUT_DATA / 'control_days.csv'}  ({len(ctl)} chain-days, "
          f"{int(ctl['chain_day'].sum())} of them the chain day itself)")

    exact = man.groupby("recurrence_key")["chain_id"].apply(list)
    n_exact = sum(1 for v in exact if len(v) > 1)
    print(f"\nrecurrence - exact identical cast: {n_exact} group(s)")
    print("recurrence - fuzzy clusters (Jaccard >= 0.5 on the cast):")
    for cl, g in man.groupby("cast_cluster"):
        shared = set.intersection(*[set(m.split("|")) for m in g["members"]])
        print(f"  {cl}: {len(g)} chain(s)  {g['direction'].iloc[0]} "
              f"track {g['tunnel_track'].iloc[0]}  "
              f"{g['start_time'].min()}-{g['start_time'].max()}")
        print(f"      dates: {list(g['iso_date'])}")
        print(f"      always present: {sorted(shared)}")


if __name__ == "__main__":
    main()
