"""Hard-route Nord-Midi trains on ConstrainedRouter, then build SUMO scenario."""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from xml.etree import ElementTree as ET

import pandas as pd

from .ConstrainedRouter import ConstrainedRouter
from .GeoUtils import GeoUtils

T0 = 7 * 3600 + 55 * 60
SPEED = 50 / 3.6
PLATFORM_SPEED = 40 / 3.6
TRAIN_LENGTH = 150.0
ACCEL = 0.4
DECEL = 0.8
INF = 10**12
STOP_HALF = 90.0
PASSAGES_FILE = "passages.xml"

STATIONS = {
    215: "BRUXELLES-CENTRAL",
    216: "BRUXELLES-CONGRES",
    217: "BRUXELLES-CHAPELLE",
    220: "BRUXELLES-MIDI",
    221: "BRUXELLES-NORD",
}

FALLBACK = {
    (221, "N2S"): 6, (221, "S2N"): 5,
    (216, "N2S"): 2, (216, "S2N"): 3,
    (215, "N2S"): 2, (215, "S2N"): 3,
    (217, "N2S"): 2, (217, "S2N"): 3,
    (220, "N2S"): 12, (220, "S2N"): 13,
}

PK_KM = {220: 0.000, 217: 1.129, 215: 1.829, 216: 2.729, 221: 3.820}


def hop_speeds_from_compare(
    compare: pd.DataFrame,
    hop_len_m: dict[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], float]:
    c = compare.sort_values(["trip_id", "seq"])
    h = c.copy()
    h["p_st"] = h.groupby("trip_id").station_id.shift()
    h["p_dep"] = h.groupby("trip_id").plan_dep_s.shift()
    h = h.dropna(subset=["p_st"])
    h["frm"] = h.p_st.astype(int)
    h["to"] = h.station_id.astype(int)
    h["dt"] = h.plan_arr_s - h.p_dep
    h = h[h.dt > 0]
    if hop_len_m:
        h["m"] = [hop_len_m.get((int(a), int(b)), float("nan")) for a, b in zip(h.frm, h.to)]
        h = h[h.m.notna() & (h.m > 1)]
        h["kmh"] = 3.6 * h.m / h.dt
    else:
        h["km"] = (h["frm"].map(PK_KM) - h["to"].map(PK_KM)).abs()
        h = h[h.km > 0]
        h["kmh"] = 3600.0 * h.km / h.dt
    out: dict[tuple[int, int], float] = {}
    for (a, b), g in h.groupby(["frm", "to"]):
        q1, q3 = g.kmh.quantile([0.25, 0.75])
        iqr = q3 - q1
        inn = g[(g.kmh >= q1 - 1.5 * iqr) & (g.kmh <= q3 + 1.5 * iqr)]
        if not len(inn):
            inn = g
        out[(int(a), int(b))] = float(inn.kmh.mean() / 3.6)
    return out


def hop_lengths_from_day(day_dir: str | Path, compare: pd.DataFrame | None = None) -> dict[tuple[int, int], float]:
    day = Path(day_dir)
    el: dict[str, float] = {}
    with open(day / "ns.net.xml") as f:
        for line in f:
            m = re.search(r'<lane id="(t\d+[FR])_0"[^>]*length="([0-9.]+)"', line)
            if m:
                el[m.group(1)] = float(m.group(2))
    stops: dict[tuple, tuple[str, float]] = {}
    for _, s in ET.iterparse(day / "stops.add.xml", events=("end",)):
        if s.tag != "trainStop":
            continue
        parts = s.get("id").split("_")
        st, plat, direction = int(parts[0][2:]), int(parts[1]), parts[2]
        edge = s.get("lane").rsplit("_", 1)[0]
        pos = 0.5 * (float(s.get("startPos")) + float(s.get("endPos")))
        stops[(st, plat, direction)] = (edge, pos)
        s.clear()
    vehs: dict[str, list[str]] = {}
    for _, v in ET.iterparse(day / "routes.rou.xml", events=("end",)):
        if v.tag != "vehicle":
            continue
        vid = v.get("id")
        for ch in v:
            if ch.tag == "route":
                vehs[vid] = ch.get("edges").split()
        v.clear()
    if compare is None:
        compare = pd.read_csv(day / "compare.csv", dtype={"train_no": str})
    compare = compare.sort_values(["trip_id", "seq"])

    def dist_along(edges, e0, p0, e1, p1):
        try:
            i0 = edges.index(e0)
            i1 = edges.index(e1, i0)
        except ValueError:
            return None
        if i0 == i1:
            return abs(p1 - p0)
        d = el.get(e0, 0) - p0
        for e in edges[i0 + 1:i1]:
            d += el.get(e, 0)
        d += p1
        return d

    acc: dict[tuple[int, int], list[float]] = defaultdict(list)
    for _, g in compare.groupby("trip_id"):
        g = g.sort_values("seq")
        tn, direction = str(g.iloc[0].train_no), g.iloc[0].direction
        edges = vehs.get(f"T{tn}_{direction}")
        if not edges:
            continue
        prev = None
        for _, r in g.iterrows():
            info = stops.get((int(r.station_id), int(r.platform), r.direction))
            if info is None:
                prev = None
                continue
            if prev is not None:
                d = dist_along(edges, prev[0], prev[1], info[0], info[1])
                if d is not None and d > 1:
                    acc[(prev[2], int(r.station_id))].append(d)
            prev = (info[0], info[1], int(r.station_id))
    return {k: float(median(v)) for k, v in acc.items() if v}


def parse_path(value):
    """Track geometry, as JSON when possible (~10x faster than literal_eval)."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return ast.literal_eval(value)


def hms(t) -> int:
    if hasattr(t, "hour"):
        return int(t.hour) * 3600 + int(t.minute) * 60 + int(getattr(t, "second", 0) or 0)
    h, m, s = map(int, str(t).split(":")[:3])
    return h * 3600 + m * 60 + s


def build_scenario(
    tracks: pd.DataFrame,
    assigned: pd.DataFrame,
    switches: pd.DataFrame,
    router: ConstrainedRouter,
    forbidden_df: pd.DataFrame,
    timetable: pd.DataFrame,
    out_dir: str | Path,
    t0: int = T0,
    speed: float = SPEED,
    platform_speed: float = PLATFORM_SPEED,
    train_length: float = TRAIN_LENGTH,
    accel: float = ACCEL,
    decel: float = DECEL,
    hop_speeds: dict[tuple[int, int], float] | None = None,
) -> dict:
    """Hard-route timetable trains and write SUMO XML files into out_dir.

    timetable columns: trip_id, train_no, seq, direction, station_id, platform, arrival, departure
    arrival/departure: 'HH:MM:SS' or datetime-like
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tracks = tracks.copy()
    tracks["ID"] = tracks["ID"].astype(int)
    tracks["Departure_switch"] = tracks["Departure_switch"].astype(int)
    tracks["Arrival_switch"] = tracks["Arrival_switch"].astype(int)

    pair_to_tid = {}
    tid_info = {}
    for tid, dep, arr, length, raw_path in zip(
        tracks["ID"], tracks["Departure_switch"], tracks["Arrival_switch"],
        tracks["Length_m"], tracks["Path"],
    ):
        tid, dep, arr = int(tid), int(dep), int(arr)
        path = parse_path(raw_path)
        pair_to_tid[(dep, arr)] = tid
        tid_info[tid] = {
            "dep": dep, "arr": arr, "length": float(length),
            "path": path, "south_F": path[-1][0] < path[0][0],
        }

    assigned = assigned.copy()
    assigned = assigned[assigned["Station_ID"].notna() & assigned["sncb_platform"].notna()]
    assigned["Station_ID"] = assigned["Station_ID"].astype(int)
    assigned = assigned[assigned["Station_ID"].isin(STATIONS)]
    assigned["sncb_platform"] = pd.to_numeric(assigned["sncb_platform"], errors="coerce")
    assigned = assigned[assigned["sncb_platform"].notna()]
    assigned["sncb_platform"] = assigned["sncb_platform"].astype(int)
    plat_track = {(int(r.Station_ID), int(r.sncb_platform)): int(r.track_id)
                  for _, r in assigned.iterrows()}
    if "position_along_track_m" in assigned.columns:
        plat_pos = {(int(r.Station_ID), int(r.sncb_platform)): float(r.position_along_track_m)
                    for _, r in assigned.iterrows()}
    else:
        station_mids: dict[int, list[list[float]]] = {}
        for (sid, _), tid in plat_track.items():
            if tid not in tid_info:
                continue
            info = tid_info[tid]
            mid = GeoUtils.pointAlongPath(info["path"], info["length"] / 2.0)
            station_mids.setdefault(sid, []).append(mid)

        station_ref = {}
        for sid, mids in station_mids.items():
            lats = sorted(m[0] for m in mids)
            lons = sorted(m[1] for m in mids)
            n = len(mids)
            station_ref[sid] = [lats[n // 2], lons[n // 2]]

        plat_pos = {}
        for (sid, plat), tid in plat_track.items():
            if tid not in tid_info or sid not in station_ref:
                continue
            info = tid_info[tid]
            pos = GeoUtils.projectOnPath(info["path"], station_ref[sid])
            if info["length"] > 2 * STOP_HALF:
                pos = max(STOP_HALF, min(info["length"] - STOP_HALF, pos))
            else:
                pos = info["length"] / 2.0
            plat_pos[(sid, plat)] = pos

    path_cache = {}

    def get_switches(src, dst):
        key = (src, dst)
        if key not in path_cache:
            path_cache[key] = router.shortestPath(src, dst)
        return path_cache[key]

    def switches_to_edges(sws):
        edges = []
        for a, b in zip(sws, sws[1:]):
            if (a, b) in pair_to_tid:
                edges.append(f"t{pair_to_tid[(a, b)]}F")
            elif (b, a) in pair_to_tid:
                edges.append(f"t{pair_to_tid[(b, a)]}R")
            else:
                raise KeyError(f"No track for oriented ({a},{b})")
        return edges

    def segment(tid_a, use_F_a, tid_b, use_F_b, forbid_tids=None):
        ia, ib = tid_info[tid_a], tid_info[tid_b]
        exit_a = ia["arr"] if use_F_a else ia["dep"]
        entry_b = ib["dep"] if use_F_b else ib["arr"]
        want_a = f"t{tid_a}{'F' if use_F_a else 'R'}"
        want_b = f"t{tid_b}{'F' if use_F_b else 'R'}"
        if tid_a == tid_b:
            return ([], 0.0) if use_F_a == use_F_b else None
        if exit_a == entry_b:
            return [], 0.0
        sws = get_switches(exit_a, entry_b)
        if sws is None:
            return None
        edges = switches_to_edges(sws)
        if forbid_tids:
            other = {str(t) for t in forbid_tids} - {str(tid_a), str(tid_b)}
            if any(e[1:-1] in other for e in edges):
                return None
        for e in edges:
            if e[1:-1] == str(tid_a) and e != want_a:
                return None
            if e[1:-1] == str(tid_b) and e != want_b:
                return None
        while edges and edges[0] == want_a:
            edges = edges[1:]
        while edges and edges[-1] == want_b:
            edges = edges[:-1]
        if any(e[1:-1] in (str(tid_a), str(tid_b)) for e in edges):
            return None
        length = sum(
            router.oriented[(a, b)]["length"]
            for a, b in zip(sws, sws[1:])
            if (a, b) in router.oriented
        )
        if length > 20000:
            return None
        return edges, length

    def resolve_platform(sid, plat, direction):
        try:
            plat = int(float(plat))
        except (TypeError, ValueError):
            plat = None
        if (sid, plat) not in plat_track:
            plat = FALLBACK[(sid, direction)]
        return plat, plat_track[(sid, plat)]

    def orient_train(tids, direction, optional_skip=None):
        n = len(tids)
        pref = []
        for tid in tids:
            south_F = tid_info[tid]["south_F"]
            pref.append(south_F if direction == "N2S" else not south_F)

        def solve(active):
            m = len(active)
            atids = [tids[i] for i in active]
            acost = [{True: INF, False: INF} for _ in range(m)]
            aback = [{True: None, False: None} for _ in range(m)]
            for o in (True, False):
                acost[0][o] = 0 if o == pref[active[0]] else 10
            for i in range(m - 1):
                for oa in (True, False):
                    if acost[i][oa] >= INF:
                        continue
                    for ob in (True, False):
                        res = segment(atids[i], oa, atids[i + 1], ob, forbid_tids=set(atids))
                        if res is None:
                            res = segment(atids[i], oa, atids[i + 1], ob,
                                          forbid_tids={atids[i], atids[i + 1]})
                        if res is None:
                            continue
                        mid, length = res
                        c = acost[i][oa] + length + (0 if ob == pref[active[i + 1]] else 50)
                        if c < acost[i + 1][ob]:
                            acost[i + 1][ob] = c
                            aback[i + 1][ob] = (oa, mid)
            best_o = True if acost[m - 1][True] <= acost[m - 1][False] else False
            if acost[m - 1][best_o] >= INF:
                return None
            orients_a = [None] * m
            mids_a = [None] * (m - 1)
            orients_a[m - 1] = best_o
            for i in range(m - 1, 0, -1):
                oa, mid = aback[i][orients_a[i]]
                orients_a[i - 1] = oa
                mids_a[i - 1] = mid
            return active, orients_a, mids_a

        full = list(range(n))
        got = solve(full)
        if got:
            return got
        if optional_skip:
            active = [i for i in full if i not in optional_skip]
            if len(active) >= 2:
                got = solve(active)
                if got:
                    return got
        return solve([0, n - 1])

    tt = timetable.copy()
    tt["seq"] = tt["seq"].astype(int)
    if "train_no" not in tt.columns:
        tt["train_no"] = tt["trip_id"].astype(str).str.split(":").str[-1]
    keep = tt.sort_values("trip_id").drop_duplicates("train_no")["trip_id"]
    tt = tt[tt["trip_id"].isin(keep)].sort_values(["trip_id", "seq"])

    vehicles = []
    used_tids = set()
    failed = []
    for trip_id, g in tt.groupby("trip_id", sort=False):
        g = g.sort_values("seq")
        direction = g.iloc[0]["direction"]
        train_no = str(g.iloc[0]["train_no"])
        stops_meta = []
        tids = []
        bad_train = False
        for _, r in g.iterrows():
            sid = int(r["station_id"])
            try:
                plat, tid = resolve_platform(sid, r["platform"], direction)
            except KeyError:
                bad_train = True
                break
            if tid not in tid_info:
                bad_train = True
                break
            tids.append(tid)
            used_tids.add(tid)
            stops_meta.append({
                "sid": sid, "plat": plat, "tid": tid,
                "arr": hms(r["arrival"]) - t0, "dep": hms(r["departure"]) - t0,
                "pos": plat_pos[(sid, plat)],
            })
        if bad_train:
            failed.append(train_no)
            continue
        optional = {i for i, s in enumerate(stops_meta) if s["sid"] in (216, 217)}
        oriented = orient_train(tids, direction, optional_skip=optional)
        if oriented is None:
            failed.append(train_no)
            continue
        active, orients, mids = oriented
        stops_meta = [stops_meta[i] for i in active]
        edges = []
        bad = False
        for i, s in enumerate(stops_meta):
            use_F = orients[i]
            s["use_F"] = use_F
            s["edge"] = f"t{s['tid']}{'F' if use_F else 'R'}"
            s["stop_id"] = f"st{s['sid']}_{s['plat']}_{direction}_{'F' if use_F else 'R'}"
            info = tid_info[s["tid"]]
            p = s["pos"] if use_F else info["length"] - s["pos"]
            s["center"] = p
            s["det_id"] = f"det{s['sid']}_{s['plat']}_{'F' if use_F else 'R'}"
            s["start"] = max(0, p - STOP_HALF)
            s["end"] = min(info["length"], p + STOP_HALF)
            if i == 0:
                edges.append(s["edge"])
            else:
                for e in mids[i - 1]:
                    used_tids.add(int(e[1:-1]))
                    if edges[-1][1:-1] == e[1:-1] and edges[-1] != e:
                        bad = True
                        break
                    if edges[-1] != e:
                        edges.append(e)
                if bad:
                    break
                if edges[-1] != s["edge"]:
                    if edges[-1][1:-1] == s["edge"][1:-1]:
                        bad = True
                        break
                    edges.append(s["edge"])
        if bad:
            failed.append(train_no)
            continue
        if any(edges[i][1:-1] == edges[i + 1][1:-1] and edges[i] != edges[i + 1]
               for i in range(len(edges) - 1)):
            failed.append(train_no)
            continue
        pos = 0
        ok_stops = True
        for s in stops_meta:
            try:
                pos = edges.index(s["edge"], pos) + 1
            except ValueError:
                ok_stops = False
                break
        if not ok_stops:
            failed.append(train_no)
            continue
        depart = max(0, stops_meta[0]["arr"])
        vehicles.append((depart, train_no, direction, edges, stops_meta))

    print(f"Trains hard-routes : {len(vehicles)} (echec: {len(failed)}) {failed[:20]}")
    print(f"Tracks utilises : {len(used_tids)}")

    used_switches = set()
    for tid in list(used_tids):
        used_switches.add(tid_info[tid]["dep"])
        used_switches.add(tid_info[tid]["arr"])
    for tid, dep, arr in zip(
        tracks["ID"], tracks["Departure_switch"], tracks["Arrival_switch"]
    ):
        if dep in used_switches and arr in used_switches:
            used_tids.add(int(tid))
    print(f"Tracks apres expansion : {len(used_tids)}")

    switches = switches.copy()
    switches["ID"] = switches["ID"].astype(int)
    sw_xy = dict(zip(switches["ID"], zip(switches["Y"], switches["X"])))

    forbidden = {
        (int(r["from_switch"]), int(r["via_switch"]), int(r["to_switch"]))
        for _, r in forbidden_df.iterrows()
    }

    sub = tracks[tracks["ID"].isin(used_tids)]
    used_nodes = set(sub["Departure_switch"]) | set(sub["Arrival_switch"])
    platform_edges = {s["edge"] for _, _, _, _, stops in vehicles for s in stops}

    edge_speed: dict[str, float] = {}
    if hop_speeds:
        votes: dict[str, list[float]] = defaultdict(list)
        for _, _, _, edges, stops_meta in vehicles:
            for a, b in zip(stops_meta, stops_meta[1:]):
                spd = hop_speeds.get((a["sid"], b["sid"]))
                if spd is None:
                    continue
                try:
                    i0 = edges.index(a["edge"])
                    i1 = edges.index(b["edge"], i0)
                except ValueError:
                    continue
                for e in edges[i0:i1 + 1]:
                    votes[e].append(spd)
        edge_speed = {e: float(median(v)) for e, v in votes.items()}

    with open(out / "ns.nod.xml", "w") as f:
        f.write("<nodes>\n")
        for nid in sorted(used_nodes):
            lon, lat = sw_xy[nid]
            f.write(f'    <node id="n{nid}" x="{lon}" y="{lat}" type="rail_signal"/>\n')
        f.write("</nodes>\n")

    with open(out / "ns.edg.xml", "w") as f:
        f.write("<edges>\n")
        for _, r in sub.iterrows():
            tid, dep, arr = int(r["ID"]), int(r["Departure_switch"]), int(r["Arrival_switch"])
            path = tid_info[tid]["path"]
            shape_f = " ".join(f"{p[1]},{p[0]}" for p in path)
            shape_r = " ".join(f"{p[1]},{p[0]}" for p in reversed(path))
            eid_f, eid_r = f"t{tid}F", f"t{tid}R"
            spd_f = edge_speed.get(eid_f, platform_speed if eid_f in platform_edges else speed)
            spd_r = edge_speed.get(eid_r, platform_speed if eid_r in platform_edges else speed)
            base = 'numLanes="1" allow="rail" spreadType="center"'
            f.write(f'    <edge id="t{tid}F" from="n{dep}" to="n{arr}" {base} speed="{spd_f:.3f}" shape="{shape_f}"/>\n')
            f.write(f'    <edge id="t{tid}R" from="n{arr}" to="n{dep}" {base} speed="{spd_r:.3f}" shape="{shape_r}"/>\n')
        f.write("</edges>\n")

    incoming, outgoing = {}, {}
    for _, r in sub.iterrows():
        tid, dep, arr = int(r["ID"]), int(r["Departure_switch"]), int(r["Arrival_switch"])
        incoming.setdefault(arr, []).append((f"t{tid}F", tid, dep))
        incoming.setdefault(dep, []).append((f"t{tid}R", tid, arr))
        outgoing.setdefault(dep, []).append((f"t{tid}F", tid, arr))
        outgoing.setdefault(arr, []).append((f"t{tid}R", tid, dep))

    n_conn = 0
    with open(out / "ns.con.xml", "w") as f:
        f.write("<connections>\n")
        for node in sorted(used_nodes):
            for e_in, t_in, from_sw in incoming.get(node, []):
                for e_out, t_out, to_sw in outgoing.get(node, []):
                    if t_in == t_out:
                        continue
                    if (from_sw, node, to_sw) in forbidden:
                        continue
                    f.write(f'    <connection from="{e_in}" to="{e_out}" fromLane="0" toLane="0"/>\n')
                    n_conn += 1
        f.write("</connections>\n")
    print(f"Connections : {n_conn}")

    stop_lines = ["<additional>"]
    seen_stops = set()
    seen_dets = set()
    for _, _, _, _, stops_meta in vehicles:
        for s in stops_meta:
            if s["stop_id"] not in seen_stops:
                seen_stops.add(s["stop_id"])
                stop_lines.append(
                    f'    <trainStop id="{s["stop_id"]}" lane="{s["edge"]}_0" '
                    f'startPos="{s["start"]:.1f}" endPos="{s["end"]:.1f}" friendlyPos="true" '
                    f'name="{STATIONS[s["sid"]]} voie {s["plat"]}"/>')
            # Non-stopping calls leave no stopinfo : a detector times them instead.
            if s["det_id"] not in seen_dets:
                seen_dets.add(s["det_id"])
                stop_lines.append(
                    f'    <instantInductionLoop id="{s["det_id"]}" lane="{s["edge"]}_0" '
                    f'pos="{s["center"]:.1f}" friendlyPos="true" file="{PASSAGES_FILE}"/>')
    stop_lines.append("</additional>")
    (out / "stops.add.xml").write_text("\n".join(stop_lines))

    vehicles.sort()
    lines = [
        "<routes>",
        f'    <vType id="train" vClass="rail" carFollowModel="Rail" '
        f'length="{train_length:.0f}" maxSpeed="{speed:.2f}" '
        f'accel="{accel}" decel="{decel}"/>',
    ]
    for depart, train_no, direction, edges, stops_meta in vehicles:
        color = "0,0.8,0" if direction == "N2S" else "0.9,0.2,0.2"
        depart = max(0, int(depart))
        first = stops_meta[0]
        dspd = "0" if int(first["dep"]) > int(first["arr"]) else "max"
        lines.append(
            f'    <vehicle id="T{train_no}_{direction}" type="train" depart="{depart}" '
            f'departPos="{first["center"]:.1f}" departSpeed="{dspd}" color="{color}">')
        lines.append(f'        <route edges="{" ".join(edges)}"/>')
        prev_until = depart
        for s in stops_meta:
            if int(s["dep"]) <= int(s["arr"]):
                continue
            until = max(int(s["dep"]), prev_until + 1, 0)
            lines.append(f'        <stop busStop="{s["stop_id"]}" until="{until}"/>')
            prev_until = until
        lines.append("    </vehicle>")
    lines.append("</routes>")
    (out / "routes.rou.xml").write_text("\n".join(lines))
    print(f"stops : {len(seen_stops)} | routes -> {out}")

    return {
        "vehicles": vehicles,
        "failed": failed,
        "used_tids": used_tids,
        "out_dir": out,
    }
