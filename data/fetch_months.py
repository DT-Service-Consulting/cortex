"""Download remaining months, extract the junction portions, discard the raw.

Disk is tight (4.6 GB free, and %TEMP% is on the same volume), so this never
accumulates raw files: each month is downloaded to a scratch dir, reduced to the
junction subsets, and the 330 MB raw is deleted before the next month starts.
Peak transient footprint is one raw file.

Per month it writes:
  junction/junction_traversing_YYYYMM.csv   long form, ~28 MB
  junction/traversals_YYYYMM.csv            wide form, ~10 MB

Usage: python data/fetch_months.py [YYYYMM ...]     (default: all missing months)
"""
from __future__ import annotations

import csv
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "junction"
SCRATCH = Path(os.environ.get("TEMP", "/tmp")) / "cortex_raw"
URL = ("https://fr.ftp.opendatasoft.com/infrabel/PunctualityHistory/"
       "Data_raw_punctuality_{month}.csv")

JUNCTION = {"215", "216", "217", "220", "221"}
NAMES = {"215": "central", "216": "congres", "217": "chapelle",
         "220": "midi", "221": "nord"}
ORDER = {"S2N": ["220", "217", "215", "216", "221"],
         "N2S": ["221", "216", "215", "217", "220"]}
ENDS = {"220", "221"}


def secs(t: str) -> int:
    t = (t or "").strip()
    if not t:
        return -1
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def to_int(v: str):
    v = (v or "").strip()
    try:
        return int(v)
    except ValueError:
        return None


def download(month: str, attempts: int = 3) -> Path:
    """Download a month, verifying the byte count against Content-Length.

    The stream can end early without raising - that silently yields a partial
    month (observed: 342 MB file arriving as 1 MB, covering one day). Always
    check the length and retry rather than trusting a clean loop exit.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    dest = SCRATCH / f"raw_{month}.csv"
    url = URL.format(month=month)

    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        n = 0
        with urllib.request.urlopen(req, timeout=300) as r, dest.open("wb") as f:
            expected = int(r.headers.get("Content-Length") or 0)
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                n += len(b)
        if expected and n < expected:
            print(f"    TRUNCATED: got {n/1e6:.1f} of {expected/1e6:.1f} MB "
                  f"(attempt {attempt}/{attempts})", flush=True)
            dest.unlink(missing_ok=True)
            if attempt == attempts:
                raise IOError(f"{month}: still short after {attempts} attempts")
            time.sleep(5 * attempt)
            continue
        print(f"    downloaded {n / 1e6:.0f} MB", flush=True)
        return dest
    raise IOError(f"{month}: download failed")


def reduce_month(raw: Path, month: str) -> dict:
    """Stream the raw file, keep junction rows, emit long + wide traversals."""
    trips: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    total = 0
    with raw.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            total += 1
            if (row.get("PTCAR_NO") or "").strip() in JUNCTION:
                trips[(row["DATDEP"], row["TRAIN_NO"])].append(row)

    keep = {}
    for k, v in trips.items():
        pts = {x["PTCAR_NO"] for x in v}
        if len(v) == 5 and ENDS <= pts:
            by = {x["PTCAR_NO"]: x for x in v}
            d = "S2N" if v[0]["PTCAR_NO"] == "220" else "N2S"
            if [x["PTCAR_NO"] for x in v] == ORDER[d]:
                keep[k] = (d, by)

    # --- long form -------------------------------------------------------
    long_path = OUT / f"junction_traversing_{month}.csv"
    extra = ["direction", "seq", "tunnel_track"]
    order = sorted(keep, key=lambda k: (k[0], secs(keep[k][1][ORDER[keep[k][0]][0]]["REAL_TIME_DEP"]), k[1]))
    with long_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + extra)
        w.writeheader()
        for k in order:
            d, by = keep[k]
            track = by["215"]["LINE_NO_DEP"]
            for i, p in enumerate(ORDER[d], start=1):
                w.writerow({**by[p], "direction": d, "seq": i, "tunnel_track": track})

    # --- wide form -------------------------------------------------------
    cols = ["date", "train_no", "relation", "relation_direction", "direction",
            "tunnel_track", "entry_line", "exit_line"]
    for p in range(1, 6):
        cols += [f"p{p}_ptcar", f"p{p}_name", f"p{p}_planned_arr", f"p{p}_real_arr",
                 f"p{p}_delay_arr", f"p{p}_planned_dep", f"p{p}_real_dep",
                 f"p{p}_delay_dep", f"p{p}_stop_type"]
    cols += ["entry_delay", "exit_delay", "delay_gained"]

    wide_path = OUT / f"traversals_{month}.csv"
    with wide_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in order:
            d, by = keep[k]
            seq = [by[p] for p in ORDER[d]]
            first, last = seq[0], seq[-1]
            rec = {"date": k[0], "train_no": k[1],
                   "relation": first["RELATION"],
                   "relation_direction": first["RELATION_DIRECTION"],
                   "direction": d, "tunnel_track": by["215"]["LINE_NO_DEP"],
                   "entry_line": first["LINE_NO_ARR"], "exit_line": last["LINE_NO_DEP"]}
            for i, r in enumerate(seq, start=1):
                rec |= {f"p{i}_ptcar": r["PTCAR_NO"], f"p{i}_name": NAMES[r["PTCAR_NO"]],
                        f"p{i}_planned_arr": r["PLANNED_TIME_ARR"],
                        f"p{i}_real_arr": r["REAL_TIME_ARR"],
                        f"p{i}_delay_arr": r["DELAY_ARR"],
                        f"p{i}_planned_dep": r["PLANNED_TIME_DEP"],
                        f"p{i}_real_dep": r["REAL_TIME_DEP"],
                        f"p{i}_delay_dep": r["DELAY_DEP"],
                        f"p{i}_stop_type": r["THOP1_COD"]}
            e, x = to_int(first["DELAY_DEP"]), to_int(last["DELAY_ARR"])
            rec["entry_delay"] = e if e is not None else ""
            rec["exit_delay"] = x if x is not None else ""
            rec["delay_gained"] = (x - e) if (e is not None and x is not None) else ""
            w.writerow(rec)

    return {"rows": total, "trips": len(trips), "traversals": len(keep),
            "long_mb": long_path.stat().st_size / 1e6,
            "wide_mb": wide_path.stat().st_size / 1e6}


def free_gb() -> float:
    import shutil
    return shutil.disk_usage(str(HERE)).free / 1e9


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    months = sys.argv[1:] or [f"2025{m:02d}" for m in range(1, 13)]
    months = [m for m in months if not (OUT / f"traversals_{m}.csv").exists()]
    if not months:
        print("nothing to do - all requested months already extracted")
        return

    print(f"free space: {free_gb():.2f} GB")
    print(f"months to fetch: {len(months)}  {months}")
    print(f"scratch: {SCRATCH}\n")

    grand = 0
    for i, month in enumerate(months, 1):
        if free_gb() < 1.0:
            print(f"!! stopping: only {free_gb():.2f} GB free")
            break
        print(f"[{i}/{len(months)}] {month}", flush=True)
        t = time.time()
        raw = download(month)
        try:
            st = reduce_month(raw, month)
        finally:
            raw.unlink(missing_ok=True)          # never keep the 330 MB raw
        grand += st["traversals"]
        print(f"    {st['rows']:,} rows -> {st['traversals']:,} traversals "
              f"({st['long_mb']:.0f} + {st['wide_mb']:.0f} MB)  [{time.time()-t:.0f}s]",
              flush=True)

    try:
        SCRATCH.rmdir()
    except OSError:
        pass
    print(f"\ntotal traversals added: {grand:,}")
    print(f"free space now: {free_gb():.2f} GB")


if __name__ == "__main__":
    main()
