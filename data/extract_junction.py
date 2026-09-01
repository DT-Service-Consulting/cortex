"""Extract the Brussels North-South junction rows from a raw Infrabel monthly file.

The raw monthly exports are ~330 MB / ~1.9M rows each, so this streams the file
once instead of loading it. Writes the junction subset plus a small profile.

Usage: python data/extract_junction.py data/raw/Data_raw_punctuality_202501.csv
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# PTCAR ids of the North-South junction, matching STATIONS in
# core/simulation/build_hard.py and the Infrabel operational-points reference.
JUNCTION = {
    "215": "BRUXELLES-CENTRAL",
    "216": "BRUXELLES-CONGRES",
    "217": "BRUXELLES-CHAPELLE",
    "220": "BRUXELLES-MIDI",
    "221": "BRUXELLES-NORD",
}


def main(src: str) -> None:
    src_path = Path(src)
    out_dir = src_path.parent.parent / "junction"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"junction_{src_path.stem.split('_')[-1]}.csv"

    total = 0
    kept = 0
    per_station: Counter = Counter()
    per_day: Counter = Counter()
    relations: Counter = Counter()
    trains: set[str] = set()
    ptcar_names: dict[str, str] = {}
    delays: defaultdict[str, list[int]] = defaultdict(list)

    with src_path.open("r", encoding="utf-8-sig", newline="") as fin, \
         out_path.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames or [])
        writer.writeheader()
        for row in reader:
            total += 1
            ptcar = (row.get("PTCAR_NO") or "").strip()
            if ptcar not in JUNCTION:
                continue
            kept += 1
            writer.writerow(row)
            per_station[ptcar] += 1
            per_day[(row.get("DATDEP") or "").strip()] += 1
            relations[(row.get("RELATION") or "").strip()] += 1
            trains.add((row.get("TRAIN_NO") or "").strip())
            ptcar_names.setdefault(ptcar, (row.get("PTCAR_LG_NM_NL") or "").strip())
            for col in ("DELAY_ARR", "DELAY_DEP"):
                v = (row.get(col) or "").strip()
                if v:
                    try:
                        delays[col].append(int(v))
                    except ValueError:
                        pass

    print(f"rows read      : {total:,}")
    print(f"junction rows  : {kept:,}  ({kept / total * 100:.2f}%)")
    print(f"distinct trains: {len(trains):,}")
    print(f"days covered   : {len(per_day)}")
    print(f"-> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("\nper station:")
    for p, n in sorted(per_station.items()):
        print(f"  {p} {JUNCTION[p]:<20} {n:>9,}   (data name: {ptcar_names.get(p, '')})")

    print("\ndelay distribution (seconds):")
    for col, vals in delays.items():
        vals.sort()
        n = len(vals)
        q = lambda f: vals[min(n - 1, int(n * f))]
        late = sum(1 for v in vals if v > 300)
        print(f"  {col}: n={n:,} min={vals[0]} p50={q(.5)} p90={q(.9)} p99={q(.99)} "
              f"max={vals[-1]}  >5min: {late / n * 100:.1f}%")

    print("\ntop relations through the junction:")
    for r, n in relations.most_common(12):
        print(f"  {r or '(blank)':<14} {n:>8,}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "data/raw/Data_raw_punctuality_202501.csv")
