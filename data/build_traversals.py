"""Reduce the junction subset to one row per Nord<->Midi traversal.

Keeps only trains that pass through BOTH Bruxelles-Nord (221) and
Bruxelles-Midi (220), i.e. that actually traverse the North-South junction.
In January 2025 that is 29,997 of 32,613 trips; the rest touch one end only.

The output is the shape delay-propagation / causal-discovery work wants:
one sample per traversal, with the delay at each of the five junction points as
its own variable, ordered along the direction of travel.

Usage: python data/build_traversals.py [junction_csv]
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Travel order through the junction, per direction.
ORDER = {
    "S2N": ["220", "217", "215", "216", "221"],
    "N2S": ["221", "216", "215", "217", "220"],
}
NAMES = {"215": "central", "216": "congres", "217": "chapelle",
         "220": "midi", "221": "nord"}
ENDS = {"220", "221"}


def to_int(v: str) -> int | None:
    v = (v or "").strip()
    try:
        return int(v)
    except ValueError:
        return None


def secs(t: str) -> int:
    """Seconds since midnight. Hours are NOT zero-padded in this feed, so
    comparing the raw strings puts '9:57:00' after '10:03:00'."""
    t = (t or "").strip()
    if not t:
        return -1
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def main(src: str) -> None:
    src_path = Path(src)
    out_path = src_path.parent / f"traversals_{src_path.stem.split('_')[-1]}.csv"

    trips: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    with src_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            trips[(row["DATDEP"], row["TRAIN_NO"])].append(row)

    cols = ["date", "train_no", "relation", "relation_direction", "direction",
            "tunnel_track", "entry_line", "exit_line"]
    for pos, key in enumerate(["p1", "p2", "p3", "p4", "p5"], start=1):
        cols += [f"{key}_ptcar", f"{key}_name", f"{key}_planned_arr",
                 f"{key}_real_arr", f"{key}_delay_arr", f"{key}_planned_dep",
                 f"{key}_real_dep", f"{key}_delay_dep", f"{key}_stop_type"]
    cols += ["entry_delay", "exit_delay", "delay_gained"]

    out_rows = []
    skipped = Counter()
    for (date, train_no), rs in trips.items():
        stations = [r["PTCAR_NO"] for r in rs]
        if not ENDS.issubset(set(stations)):
            skipped["does_not_traverse"] += 1
            continue
        if len(rs) != 5:
            skipped["not_five_points"] += 1
            continue
        direction = "S2N" if stations[0] == "220" else "N2S"
        if stations != ORDER[direction]:
            skipped["unexpected_order"] += 1
            continue

        by_pt = {r["PTCAR_NO"]: r for r in rs}
        seq = [by_pt[p] for p in ORDER[direction]]
        first, last = seq[0], seq[-1]

        rec = {
            "date": date,
            "train_no": train_no,
            "relation": first["RELATION"],
            "relation_direction": first["RELATION_DIRECTION"],
            "direction": direction,
            # every traversal holds one tunnel track across 215/216/217
            "tunnel_track": by_pt["215"]["LINE_NO_DEP"],
            "entry_line": first["LINE_NO_ARR"],
            "exit_line": last["LINE_NO_DEP"],
        }
        for pos, r in enumerate(seq, start=1):
            k = f"p{pos}"
            rec |= {
                f"{k}_ptcar": r["PTCAR_NO"],
                f"{k}_name": NAMES[r["PTCAR_NO"]],
                f"{k}_planned_arr": r["PLANNED_TIME_ARR"],
                f"{k}_real_arr": r["REAL_TIME_ARR"],
                f"{k}_delay_arr": r["DELAY_ARR"],
                f"{k}_planned_dep": r["PLANNED_TIME_DEP"],
                f"{k}_real_dep": r["REAL_TIME_DEP"],
                f"{k}_delay_dep": r["DELAY_DEP"],
                f"{k}_stop_type": r["THOP1_COD"],
            }

        entry = to_int(first["DELAY_DEP"])
        exit_ = to_int(last["DELAY_ARR"])
        rec["entry_delay"] = entry if entry is not None else ""
        rec["exit_delay"] = exit_ if exit_ is not None else ""
        rec["delay_gained"] = (exit_ - entry) if (entry is not None and exit_ is not None) else ""
        out_rows.append(rec)

    out_rows.sort(key=lambda r: (r["date"], secs(r["p1_real_dep"]), r["train_no"]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    print(f"traversals     : {len(out_rows):,}")
    print(f"skipped        : {dict(skipped)}")
    print(f"-> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("\nby direction:", dict(Counter(r["direction"] for r in out_rows)))
    print("by tunnel track:", dict(sorted(Counter(r["tunnel_track"] for r in out_rows).items())))

    print("\ndelay gained across the junction (exit arr - entry dep, seconds):")
    for d in ("S2N", "N2S"):
        vals = sorted(r["delay_gained"] for r in out_rows
                      if r["direction"] == d and r["delay_gained"] != "")
        n = len(vals)
        q = lambda f: vals[min(n - 1, int(n * f))]
        print(f"  {d}: n={n:,} p10={q(.1)} p50={q(.5)} mean={statistics.mean(vals):.0f} "
              f"p90={q(.9)} p99={q(.99)}")

    crossings = sum(1 for r in out_rows
                    if secs(r["p1_real_dep"]) >= 0 and secs(r["p5_real_arr"]) >= 0
                    and secs(r["p1_real_dep"]) > secs(r["p5_real_arr"]))
    print(f"\nmidnight-crossing traversals: {crossings}")

    print("\nmedian arrival delay by position along travel (seconds):")
    for d in ("S2N", "N2S"):
        cells = []
        for pos in range(1, 6):
            vals = [to_int(r[f"p{pos}_delay_arr"]) for r in out_rows if r["direction"] == d]
            vals = sorted(v for v in vals if v is not None)
            name = ORDER[d][pos - 1]
            cells.append(f"{NAMES[name]}={statistics.median(vals):.0f}")
        print(f"  {d}: " + "  ->  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/junction/junction_202501.csv")
