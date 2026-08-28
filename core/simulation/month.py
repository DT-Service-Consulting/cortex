"""Run the Nord-Midi SUMO scenario day by day over a period."""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent
from xml.etree import ElementTree as ET

import pandas as pd

from .build_hard import PASSAGES_FILE, SPEED, build_scenario, hms

SIM_END = 30 * 3600
STOP_RE = re.compile(r"st(\d+)_(\d+)_(N2S|S2N)_([FR])")
DET_RE = re.compile(r"det(\d+)_(\d+)_([FR])")
VEH_RE = re.compile(r"T(.+)_(N2S|S2N)$")
KEYS = ["train_no", "direction", "station_id", "platform"]


def parse_datdep(datdep: str) -> datetime:
    return datetime.strptime(datdep.upper(), "%d%b%Y")


def format_datdep(day: datetime) -> str:
    return day.strftime("%d%b%Y").upper()


def date_range(start: str, days: int) -> list[str]:
    d0 = parse_datdep(start)
    return [format_datdep(d0 + timedelta(days=i)) for i in range(days)]


def direction_of(stations: list[int]) -> str | None:
    if 221 in stations and 220 in stations:
        return "N2S" if stations.index(221) < stations.index(220) else "S2N"
    if stations[0] == 221:
        return "N2S"
    if stations[0] == 220:
        return "S2N"
    return None


def _clock(t, base: datetime) -> str | None:
    if t is None or pd.isna(t):
        return None
    total = int((pd.Timestamp(t).to_pydatetime() - base).total_seconds())
    if total < 0:
        return None
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def build_timetable(planning: pd.DataFrame, datdep: str, clock: str = "planned") -> pd.DataFrame:
    """Corridor timetable for one DATDEP, in seconds since that day's midnight.

    clock='planned' uses PLANNED_DATETIME_* ; clock='real' uses REAL_DATETIME_*
    (falls back to planned if a real timestamp is missing).
    Station order always follows planned times.
    """
    arr_col = "REAL_DATETIME_ARR" if clock == "real" else "PLANNED_DATETIME_ARR"
    dep_col = "REAL_DATETIME_DEP" if clock == "real" else "PLANNED_DATETIME_DEP"
    if clock == "real" and arr_col not in planning.columns:
        raise KeyError("REAL_DATETIME_ARR/DEP manquants dans le planning")

    base = parse_datdep(datdep)
    pdf = planning[planning["DATDEP"] == datdep].copy()
    pdf["platform"] = pd.to_numeric(pdf["platform"], errors="coerce")
    pdf = pdf[pdf["platform"].notna()]
    pdf = pdf.sort_values(
        ["TRAIN_NO", "PLANNED_DATETIME_ARR", "PLANNED_DATETIME_DEP"], na_position="last"
    )

    rows = []
    for train_no, g in pdf.groupby("TRAIN_NO", sort=False):
        stations = g["PTCAR_ID"].astype(int).tolist()
        if len(stations) < 2:
            continue
        direction = direction_of(stations)
        if direction is None:
            continue
        trip_id = f"{datdep}:{train_no}:{direction}"
        stops = []
        for seq, (_, r) in enumerate(g.iterrows()):
            arr, dep = r[arr_col], r[dep_col]
            if arr is None or pd.isna(arr):
                arr = r.get("PLANNED_DATETIME_ARR", dep)
            if dep is None or pd.isna(dep):
                dep = r.get("PLANNED_DATETIME_DEP", arr)
            if arr is None or pd.isna(arr):
                arr = dep
            if dep is None or pd.isna(dep):
                dep = arr
            arr_s, dep_s = _clock(arr, base), _clock(dep, base)
            if not arr_s or not dep_s:
                continue
            if hms(dep_s) < hms(arr_s):
                dep_s = arr_s
            stops.append({
                "datdep": datdep,
                "trip_id": trip_id,
                "train_no": str(train_no),
                "seq": seq,
                "direction": direction,
                "station_id": int(r["PTCAR_ID"]),
                "platform": int(r["platform"]),
                "arrival": arr_s,
                "departure": dep_s,
                "clock": clock,
            })
        if len(stops) >= 2:
            rows.extend(stops)

    tt = pd.DataFrame(rows)
    if len(tt):
        tt["seq"] = tt.groupby("trip_id").cumcount()
    return tt


def write_configs(day_dir: Path, end: int = SIM_END) -> None:
    (day_dir / "ns.batch.sumocfg").write_text(dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <configuration>
            <input>
                <net-file value="ns.net.xml"/>
                <route-files value="routes.rou.xml"/>
                <additional-files value="stops.add.xml"/>
            </input>
            <output>
                <stop-output value="stop_events.xml"/>
                <tripinfo-output value="tripinfos.xml"/>
            </output>
            <time>
                <begin value="0"/>
                <end value="{end}"/>
            </time>
            <processing>
                <collision.action value="warn"/>
                <time-to-teleport value="-1"/>
            </processing>
            <report>
                <no-step-log value="true"/>
                <duration-log.statistics value="true"/>
            </report>
        </configuration>
    """))


def netconvert(day_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run([
        "netconvert",
        "-n", str(day_dir / "ns.nod.xml"),
        "-e", str(day_dir / "ns.edg.xml"),
        "-x", str(day_dir / "ns.con.xml"),
        "-o", str(day_dir / "ns.net.xml"),
        "--proj.plain-geo", "true",
        "--proj", "+proj=utm +zone=31 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
        "--no-turnarounds", "true",
    ], capture_output=True, text=True)


def read_stop_events(path: Path) -> pd.DataFrame:
    rows = []
    for _, s in ET.iterparse(str(path), events=("end",)):
        if s.tag != "stopinfo":
            continue
        m = STOP_RE.match(s.get("busStop") or "")
        if not m:
            continue
        sid, plat, direction, _ = m.groups()
        vm = VEH_RE.match(s.get("id", ""))
        rows.append({
            "train_no": vm.group(1) if vm else s.get("id", ""),
            "direction": direction,
            "station_id": int(sid),
            "platform": int(plat),
            "sim_arr_s": float(s.get("started")),
            "sim_dep_s": float(s.get("ended")),
            "stop_id": s.get("busStop"),
        })
        s.clear()
    return pd.DataFrame(rows)


def read_passages(path: Path) -> pd.DataFrame:
    """Detector crossings at platform centres, for calls served without stopping."""
    rows = []
    for _, e in ET.iterparse(str(path), events=("end",)):
        if e.tag != "instantOut" or e.get("state") != "enter":
            e.clear()
            continue
        m = DET_RE.match(e.get("id") or "")
        vm = VEH_RE.match(e.get("vehID", ""))
        if m and vm:
            sid, plat, _ = m.groups()
            rows.append({
                "train_no": vm.group(1),
                "direction": vm.group(2),
                "station_id": int(sid),
                "platform": int(plat),
                "sim_pass_s": float(e.get("time")),
            })
        e.clear()
    passages = pd.DataFrame(rows)
    if len(passages):
        passages = passages.drop_duplicates(KEYS)
    return passages


def compare(tt: pd.DataFrame, sim: pd.DataFrame, passages: pd.DataFrame) -> pd.DataFrame:
    plan = tt.copy()
    plan["train_no"] = plan["train_no"].astype(str)
    plan["station_id"] = plan["station_id"].astype(int)
    plan["platform"] = pd.to_numeric(plan["platform"], errors="coerce").astype(int)
    plan["plan_arr_s"] = plan["arrival"].map(hms)
    plan["plan_dep_s"] = plan["departure"].map(hms)

    out = plan
    for frame in (passages, sim):
        if len(frame):
            out = out.merge(frame, on=KEYS, how="left")
    for col in ("sim_pass_s", "sim_arr_s", "sim_dep_s"):
        if col not in out.columns:
            out[col] = float("nan")

    # Platform centre crossing is the common reference : it exists whether the
    # train stops or runs through. Stop events only add the departure.
    out["has_stop"] = out["sim_dep_s"].notna()
    out["sim_arr_s"] = out["sim_pass_s"].fillna(out["sim_arr_s"])
    out["sim_dep_s"] = out["sim_dep_s"].fillna(out["sim_arr_s"])
    out = out[out["sim_arr_s"].notna()]
    out["delay_arr_s"] = out["sim_arr_s"] - out["plan_arr_s"]
    out["delay_dep_s"] = out["sim_dep_s"] - out["plan_dep_s"]
    return out


def run_day(
    datdep: str,
    planning: pd.DataFrame,
    tracks: pd.DataFrame,
    assigned: pd.DataFrame,
    switches: pd.DataFrame,
    router,
    forbidden_df: pd.DataFrame,
    out_root: str | Path,
    end: int = SIM_END,
    speed: float = SPEED,
    clock: str = "planned",
    skip_done: bool = True,
    hop_speeds: dict | None = None,
) -> dict:
    day_dir = Path(out_root) / parse_datdep(datdep).strftime("%Y%m%d")
    done = day_dir / "compare.csv"
    if skip_done and done.exists():
        cached = pd.read_csv(done, dtype={"train_no": str, "datdep": str})
        return {"datdep": datdep, "compare": cached, "skipped": True}

    day_dir.mkdir(parents=True, exist_ok=True)
    tt = build_timetable(planning, datdep, clock=clock)
    if not len(tt):
        return {"datdep": datdep, "compare": pd.DataFrame(), "trips": 0}

    scenario = build_scenario(
        tracks=tracks, assigned=assigned, switches=switches,
        router=router, forbidden_df=forbidden_df, timetable=tt,
        out_dir=day_dir, t0=0, speed=speed, hop_speeds=hop_speeds,
    )
    write_configs(day_dir, end=end)

    nc = netconvert(day_dir)
    if nc.returncode != 0:
        return {"datdep": datdep, "error": "netconvert", "log": nc.stderr[-800:]}

    run = subprocess.run(
        ["sumo", "-c", str(day_dir / "ns.batch.sumocfg")], capture_output=True, text=True
    )
    if not (day_dir / "stop_events.xml").exists():
        return {"datdep": datdep, "error": "sumo", "log": (run.stderr or run.stdout)[-800:]}

    sim = read_stop_events(day_dir / "stop_events.xml")
    crossings = day_dir / PASSAGES_FILE
    passages = read_passages(crossings) if crossings.exists() else pd.DataFrame()
    cmp_df = compare(tt, sim, passages)
    cmp_df.to_csv(done, index=False)

    return {
        "datdep": datdep,
        "trips": tt["trip_id"].nunique(),
        "vehicles": len(scenario["vehicles"]),
        "failed": len(scenario["failed"]),
        "stops": len(sim),
        "passages": len(passages),
        "matched": len(cmp_df),
        "compare": cmp_df,
        "dir": day_dir,
    }


def run_period(days: list[str], out_root: str | Path, **kw) -> pd.DataFrame:
    out_root = Path(out_root)
    frames = []
    for datdep in days:
        res = run_day(datdep, out_root=out_root, **kw)
        if "error" in res:
            print(f"{datdep}  ERREUR {res['error']}: {res['log'][:200]}")
            continue
        if res.get("skipped"):
            print(f"{datdep}  deja fait ({len(res['compare'])} passages)")
        else:
            c = res["compare"]
            d = c["delay_arr_s"] if len(c) else pd.Series(dtype=float)
            print(f"{datdep}  trains {res.get('vehicles', 0)}/{res.get('trips', 0)}"
                  f" | echecs {res.get('failed', 0)}"
                  f" | passages {res.get('matched', 0)}"
                  f" (dont arrets {int(c['has_stop'].sum()) if len(c) else 0})"
                  f" | retard median {d.median() if len(d) else float('nan'):.0f}s")
        frames.append(res["compare"])

    allc = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(allc):
        allc.to_parquet(out_root / "compare_all_days.parquet", index=False)
    return allc
