"""Recover the upstream network state that the junction extraction throws away.

`fetch_months.py` reduces a 1.9M-row network-wide monthly export to the 5 junction
PTCARs and deletes the raw. That was the right call for disk, but PREDICTABILITY.md
shows what it costs: above ~5 min headway the model's Brier skill collapses to
0.02-0.04, because the residual risk is driven by things that happened to the train
*before* it reached Brussels - and those columns were filtered out at ingest.

This recovers them without keeping the raw. Two products per month, both tiny
(a few MB against 330 MB in), because everything is aggregated during the stream:

  upstream/upstream_YYYYMM.csv   one row per junction traversal: the train's own
                                 delay trajectory over its last points before entry
  upstream/netstate_YYYYMM.csv   network-wide congestion per 15-min bin: how late
                                 the whole system is at the moment of entry

The two answer different questions. `upstream_*` is "where is THIS train in its own
run" - is it shedding delay or accumulating it, and for how long. `netstate_*` is
"how bad is the network right now", which is the common-cause channel the junction
data cannot see at all. Neither is available in `data/junction/`.

Disk contract is inherited from fetch_months.py and must be kept: download to
scratch, stream once, delete the raw before the next month. Peak transient
footprint is one raw file. A locally present raw in data/raw/ is reused, not
re-downloaded.

Usage: python data/fetch_upstream.py [YYYYMM ...]     (default: all months with a
                                                       junction extraction present)
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_months import SCRATCH, download, secs  # noqa: E402

HERE = Path(__file__).resolve().parent
JUNCTION_DIR = HERE / "junction"
LOCAL_RAW = HERE / "raw"
OUT = HERE / "upstream"

JUNCTION = {"215", "216", "217", "220", "221"}
BIN_S = 900                       # 15-minute network-state bins
LATE_S = 60                       # "late" for the network-state share
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

UP_COLS = ["date", "train_no", "up_n", "up_span_s", "up_d1", "up_d2", "up_d3",
           "up_first", "up_max", "up_min", "up_mean", "up_slope", "up_worsening",
           "net_mean", "net_p_late", "net_n"]


def to_int(v):
    try:
        return int((v or "").strip())
    except ValueError:
        return None


def daykey(d: str) -> int:
    """01APR2025 -> 20250401, so rows sort across a midnight rollover."""
    try:
        return int(d[5:9]) * 10000 + MONTHS[d[2:5].upper()] * 100 + int(d[:2])
    except (KeyError, ValueError):
        return 0


def row_delay(r: dict):
    """Departure delay where the train departs, arrival delay at its last point."""
    d = to_int(r["DELAY_DEP"])
    return d if d is not None else to_int(r["DELAY_ARR"])


def row_time(r: dict) -> int:
    t = secs(r["PLANNED_TIME_DEP"])
    return t if t >= 0 else secs(r["PLANNED_TIME_ARR"])


def load_keys(month: str) -> dict:
    """(date, train_no) -> entry PTCAR, for traversals already extracted."""
    p = JUNCTION_DIR / f"traversals_{month}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} - run fetch_months.py for {month} first")
    keys = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            keys[(r["date"], r["train_no"])] = r["p1_ptcar"]
    return keys


def stream(raw: Path, keys: dict):
    """One pass: collect the tracked trains' rows and the network-state bins."""
    rows: defaultdict[tuple, list] = defaultdict(list)
    net: defaultdict[tuple, list] = defaultdict(lambda: [0, 0, 0])   # n, sum, n_late
    with raw.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            k = (r["DATDEP"], r["TRAIN_NO"])
            t, d = row_time(r), row_delay(r)
            if t >= 0 and d is not None:
                b = net[(r["DATDEP"], t // BIN_S)]
                b[0] += 1
                b[1] += d
                b[2] += d > LATE_S
            if k in keys:
                rows[k].append((daykey(r.get("PLANNED_DATE_DEP") or r["DATDEP"]), t,
                                r["PTCAR_NO"], d))
    return rows, net


def features(seq: list, entry_ptcar: str) -> dict | None:
    """Delay trajectory over the points preceding junction entry."""
    seq = sorted(seq, key=lambda x: (x[0], x[1]))
    idx = next((i for i, x in enumerate(seq) if x[2] == entry_ptcar), None)
    if idx is None:
        return None
    up = [x for x in seq[:idx] if x[2] not in JUNCTION and x[3] is not None]
    if not up:
        return None                       # junction is the train's origin
    d = [x[3] for x in up]
    back = list(reversed(d))              # back[0] = last point before entry
    n = len(d)
    return {
        "up_n": n,
        "up_span_s": seq[idx][1] - up[0][1],
        "up_d1": back[0],
        "up_d2": back[1] if n > 1 else "",
        "up_d3": back[2] if n > 2 else "",
        "up_first": d[0],
        "up_max": max(d),
        "up_min": min(d),
        "up_mean": round(sum(d) / n, 1),
        # trend over the last three points: positive = still accumulating
        "up_slope": round((back[0] - back[min(2, n - 1)]) / max(1, min(2, n - 1)), 1),
        "up_worsening": int(back[0] > back[min(2, n - 1)]),
    }


def write_month(month: str, rows: dict, net: dict, keys: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    up_path = OUT / f"upstream_{month}.csv"
    kept = 0
    with up_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UP_COLS)
        w.writeheader()
        for k, seq in rows.items():
            fe = features(seq, keys[k])
            if fe is None:
                continue
            entry_t = next((x[1] for x in sorted(seq, key=lambda x: (x[0], x[1]))
                            if x[2] == keys[k]), -1)
            b = net.get((k[0], entry_t // BIN_S)) if entry_t >= 0 else None
            fe |= {"date": k[0], "train_no": k[1],
                   "net_mean": round(b[1] / b[0], 1) if b else "",
                   "net_p_late": round(b[2] / b[0], 4) if b else "",
                   "net_n": b[0] if b else ""}
            w.writerow(fe)
            kept += 1

    net_path = OUT / f"netstate_{month}.csv"
    with net_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "bin", "n", "mean_delay", "p_late"])
        for (date, b), (n, s, late) in sorted(net.items()):
            w.writerow([date, b, n, round(s / n, 1), round(late / n, 4)])

    return {"traversals": len(keys), "with_upstream": kept, "bins": len(net),
            "mb": (up_path.stat().st_size + net_path.stat().st_size) / 1e6}


def free_gb() -> float:
    return shutil.disk_usage(str(HERE)).free / 1e9


def main() -> None:
    months = sys.argv[1:] or sorted(
        p.stem.split("_")[-1] for p in JUNCTION_DIR.glob("traversals_*.csv"))
    months = [m for m in months if not (OUT / f"upstream_{m}.csv").exists()]
    if not months:
        print("nothing to do - all months already have upstream features")
        return

    print(f"free space: {free_gb():.2f} GB")
    print(f"months: {len(months)}  {months}\n", flush=True)

    for i, month in enumerate(months, 1):
        if free_gb() < 1.0:
            print(f"!! stopping: only {free_gb():.2f} GB free")
            break
        print(f"[{i}/{len(months)}] {month}", flush=True)
        t = time.time()
        local = LOCAL_RAW / f"Data_raw_punctuality_{month}.csv"
        raw, borrowed = (local, True) if local.exists() else (download(month), False)
        try:
            keys = load_keys(month)
            rows, net = stream(raw, keys)
            st = write_month(month, rows, net, keys)
        finally:
            if not borrowed:
                raw.unlink(missing_ok=True)      # never keep the 330 MB raw
        print(f"    {st['with_upstream']:,}/{st['traversals']:,} traversals have "
              f"upstream history, {st['bins']:,} net bins ({st['mb']:.1f} MB) "
              f"[{time.time() - t:.0f}s]", flush=True)

    if SCRATCH.exists():
        try:
            SCRATCH.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
