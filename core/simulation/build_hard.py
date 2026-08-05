"""Hard-route Nord-Midi trains on ConstrainedRouter, then build SUMO scenario."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from .ConstrainedRouter import ConstrainedRouter

T0 = 7 * 3600 + 55 * 60
SPEED = 22.0
INF = 10**12

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
    for _, r in tracks.iterrows():
        tid = int(r["ID"])
        dep, arr = int(r["Departure_switch"]), int(r["Arrival_switch"])
        path = ast.literal_eval(r["Path"]) if isinstance(r["Path"], str) else r["Path"]
        pair_to_tid[(dep, arr)] = tid
        tid_info[tid] = {
            "dep": dep, "arr": arr, "length": float(r["Length_m"]),
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
        plat_pos = {
            (sid, plat): tid_info[tid]["length"] / 2.0
            for (sid, plat), tid in plat_track.items()
            if tid in tid_info
        }

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
            s["start"] = max(0, p - 90)
            s["end"] = min(info["length"], p + 90)
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
        depart = max(0, stops_meta[0]["arr"] - 60)
        vehicles.append((depart, train_no, direction, edges, stops_meta))

    print(f"Trains hard-routes : {len(vehicles)} (echec: {len(failed)}) {failed[:20]}")
    print(f"Tracks utilises : {len(used_tids)}")

    used_switches = set()
    for tid in list(used_tids):
        used_switches.add(tid_info[tid]["dep"])
        used_switches.add(tid_info[tid]["arr"])
    for _, r in tracks.iterrows():
        if r["Departure_switch"] in used_switches and r["Arrival_switch"] in used_switches:
            used_tids.add(int(r["ID"]))
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
            path = ast.literal_eval(r["Path"]) if isinstance(r["Path"], str) else r["Path"]
            shape_f = " ".join(f"{p[1]},{p[0]}" for p in path)
            shape_r = " ".join(f"{p[1]},{p[0]}" for p in reversed(path))
            common = f'numLanes="1" speed="{SPEED}" allow="rail" spreadType="center"'
            f.write(f'    <edge id="t{tid}F" from="n{dep}" to="n{arr}" {common} shape="{shape_f}"/>\n')
            f.write(f'    <edge id="t{tid}R" from="n{arr}" to="n{dep}" {common} shape="{shape_r}"/>\n')
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
    for _, _, _, _, stops_meta in vehicles:
        for s in stops_meta:
            if s["stop_id"] in seen_stops:
                continue
            seen_stops.add(s["stop_id"])
            stop_lines.append(
                f'    <trainStop id="{s["stop_id"]}" lane="{s["edge"]}_0" '
                f'startPos="{s["start"]:.1f}" endPos="{s["end"]:.1f}" friendlyPos="true" '
                f'name="{STATIONS[s["sid"]]} voie {s["plat"]}"/>')
    stop_lines.append("</additional>")
    (out / "stops.add.xml").write_text("\n".join(stop_lines))

    vehicles.sort()
    lines = [
        "<routes>",
        '    <vType id="train" vClass="rail" length="40" maxSpeed="33" accel="0.6" decel="0.9"/>',
    ]
    for depart, train_no, direction, edges, stops_meta in vehicles:
        color = "0,0.8,0" if direction == "N2S" else "0.9,0.2,0.2"
        depart = max(0, int(depart))
        lines.append(
            f'    <vehicle id="T{train_no}_{direction}" type="train" depart="{depart}" color="{color}">')
        lines.append(f'        <route edges="{" ".join(edges)}"/>')
        prev_until = depart
        for i, s in enumerate(stops_meta):
            if i == len(stops_meta) - 1:
                lines.append(f'        <stop busStop="{s["stop_id"]}" duration="60"/>')
            else:
                until = max(int(s["dep"]), int(s["arr"]) + 20, prev_until + 1, 0)
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
