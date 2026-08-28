from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import networkx as nx
import pandas as pd

from .build_hard import hms

CORRIDOR = {
    "N2S": [221, 216, 215, 217, 220],
    "S2N": [220, 217, 215, 216, 221],
}

RUN_S = {
    (221, 216): 100,
    (216, 215): 100,
    (215, 217): 90,
    (217, 220): 120,
    (220, 217): 120,
    (217, 215): 90,
    (215, 216): 100,
    (216, 221): 100,
}

NORMAL_DWELL = {221: 50, 216: 35, 215: 45, 217: 35, 220: 50}

N2S_JAMBES = [
    {221: 12, 216: 6, 215: 6, 217: 6, 220: 21},
    {221: 4, 216: 2, 215: 2, 217: 2, 220: 9},
    {221: 9, 216: 4, 215: 4, 217: 4, 220: 15},
]
S2N_JAMBES = [
    {220: 19, 217: 5, 215: 5, 216: 5, 221: 11},
    {220: 3, 217: 1, 215: 1, 216: 1, 221: 3},
    {220: 13, 217: 3, 215: 3, 216: 3, 221: 7},
]


def jambe_spec(jambe: int) -> tuple[str, dict[int, int]]:
    if jambe < len(N2S_JAMBES):
        return "N2S", N2S_JAMBES[jambe]
    return "S2N", S2N_JAMBES[(jambe - len(N2S_JAMBES)) % len(S2N_JAMBES)]


def clock(sec: int) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


@dataclass
class CausalTrain:
    train_no: str
    role: str
    direction: str = "N2S"
    parents: list[str] = field(default_factory=list)
    jambe: int = 0


@dataclass
class CausalGraph:
    trains: list[CausalTrain]
    conflict_station: int = 220
    extra_dwell_s: int = 600
    gain_dwell_s: int = 20
    headway_s: int = 180
    t0_clock: int = 8 * 3600

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [(p, t.train_no) for t in self.trains for p in t.parents]

    @property
    def by_id(self) -> dict[str, CausalTrain]:
        return {t.train_no: t for t in self.trains}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edges"] = self.edges
        return d


def _bind_directions(graph: CausalGraph) -> CausalGraph:
    for t in graph.trains:
        t.direction, _ = jambe_spec(t.jambe)
    return graph


def example_graph() -> CausalGraph:
    return _bind_directions(CausalGraph(
        trains=[
            CausalTrain("1", "primary", jambe=0),
            CausalTrain("2", "secondary", parents=["1"], jambe=0),
            CausalTrain("3", "secondary", parents=["1", "2"], jambe=0),
            CausalTrain("4", "gain", jambe=1),
            CausalTrain("5", "control", jambe=2),
        ]
    ))


def random_graph(
    n_primary: int = 1,
    chain_len: int = 3,
    n_gain: int = 1,
    n_control: int = 1,
    seed: int = 0,
    extra_dwell_s: int = 600,
    gain_dwell_s: int = 20,
    headway_s: int = 180,
    conflict_station: int = 220,
    p_long_edge: float = 0.7,
) -> CausalGraph:
    rng = random.Random(seed)
    trains: list[CausalTrain] = []
    n = 1
    for jambe in range(n_primary):
        ids = []
        for i in range(max(2, chain_len)):
            tid = str(n)
            n += 1
            if i == 0:
                trains.append(CausalTrain(tid, "primary", jambe=jambe))
            else:
                trains.append(CausalTrain(tid, "secondary", parents=[ids[-1]], jambe=jambe))
            ids.append(tid)
        if len(ids) >= 3 and rng.random() < p_long_edge:
            trains[-1].parents.append(ids[0])
    jambe = n_primary
    for _ in range(n_gain):
        trains.append(CausalTrain(str(n), "gain", jambe=jambe))
        n += 1
        jambe += 1
    for _ in range(n_control):
        trains.append(CausalTrain(str(n), "control", jambe=jambe))
        n += 1
        jambe += 1
    return _bind_directions(CausalGraph(
        trains=trains,
        conflict_station=conflict_station,
        extra_dwell_s=extra_dwell_s,
        gain_dwell_s=gain_dwell_s,
        headway_s=headway_s,
    ))


def graph_table(graph: CausalGraph) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "train_no": t.train_no,
            "role": t.role,
            "direction": t.direction,
            "jambe": t.jambe,
            "parents": ",".join(t.parents) if t.parents else "",
        }
        for t in graph.trains
    ])


def make_platform_map(n_jambes: int) -> dict:
    out = {}
    for j in range(n_jambes):
        direction, plats = jambe_spec(j)
        for sid, plat in plats.items():
            out[(int(sid), direction, j)] = int(plat)
    return out


def _topo(trains: list[CausalTrain]) -> list[CausalTrain]:
    by_id = {t.train_no: t for t in trains}
    g = nx.DiGraph()
    for t in trains:
        g.add_node(t.train_no)
        for p in t.parents:
            if p in by_id:
                g.add_edge(p, t.train_no)
    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        order = [t.train_no for t in trains]
    return [by_id[i] for i in order]


def _dwell(graph: CausalGraph, train: CausalTrain, sid: int, extra_for: set[str], apply_gain: bool) -> int:
    d = NORMAL_DWELL.get(sid, 40)
    if train.train_no in extra_for and sid == graph.conflict_station:
        d += graph.extra_dwell_s
    if apply_gain and train.role == "gain":
        d = min(d, graph.gain_dwell_s)
    return max(d, 5)


def build_timetable(
    graph: CausalGraph,
    platforms: dict,
    extra_for: set[str] | None = None,
    apply_gain: bool = False,
) -> pd.DataFrame:
    if extra_for is None:
        extra_for = {t.train_no for t in graph.trains if t.role == "primary"}
    groups: dict[tuple[str, int], list[CausalTrain]] = {}
    for train in _topo(graph.trains):
        groups.setdefault((train.direction, train.jambe), []).append(train)
    rows = []
    for (direction, jambe), group in groups.items():
        stations = CORRIDOR[direction]
        for k, train in enumerate(group):
            t = graph.t0_clock + k * graph.headway_s
            for seq, sid in enumerate(stations):
                plat = platforms[(sid, direction, jambe)]
                arr = t
                dep = arr + _dwell(graph, train, sid, extra_for, apply_gain)
                rows.append({
                    "trip_id": f"GT:{train.train_no}:{direction}",
                    "train_no": str(train.train_no),
                    "seq": seq,
                    "direction": direction,
                    "station_id": int(sid),
                    "platform": int(plat),
                    "arrival": clock(arr),
                    "departure": clock(dep),
                    "role": train.role,
                    "jambe": jambe,
                })
                if seq < len(stations) - 1:
                    t = dep + RUN_S[(sid, stations[seq + 1])]
    return pd.DataFrame(rows)


def reference_timetable(graph: CausalGraph, platforms: dict) -> pd.DataFrame:
    return build_timetable(graph, platforms, extra_for=set(), apply_gain=False)


def absurd_timetable(graph: CausalGraph, platforms: dict) -> pd.DataFrame:
    return build_timetable(graph, platforms, extra_for=None, apply_gain=True)


def delay_by_train(cmp_df: pd.DataFrame, station_id: int | None = None) -> pd.Series:
    df = cmp_df
    if station_id is not None and len(df) and "station_id" in df.columns:
        sub = df[df["station_id"] == station_id]
        if len(sub):
            df = sub
    if not len(df):
        return pd.Series(dtype=float)
    return df.groupby(df["train_no"].astype(str))["delay_arr_s"].max()


def detect_edges(
    baseline: pd.Series,
    interventions: dict[str, pd.Series],
    threshold_s: float = 60,
) -> list[tuple[str, str, float]]:
    edges = []
    for cause, delayed in interventions.items():
        delta = delayed.subtract(baseline, fill_value=0.0)
        for effect, value in delta.items():
            if str(effect) == str(cause):
                continue
            if float(value) >= threshold_s:
                edges.append((str(cause), str(effect), float(value)))
    return edges


def score_edges(truth: list[tuple[str, str]], pred: list[tuple[str, str]]):
    t = {(a, b) for a, b in truth}
    p = {(a, b) for a, b in pred}
    tp, fp, fn = t & p, p - t, t - p
    prec = len(tp) / len(p) if p else 1.0
    rec = len(tp) / len(t) if t else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": sorted(tp),
        "fp": sorted(fp),
        "fn": sorted(fn),
    }


def dwell_preview(tt: pd.DataFrame) -> pd.DataFrame:
    df = tt.copy()
    df["arr_s"] = df["arrival"].map(hms)
    df["dep_s"] = df["departure"].map(hms)
    df["dwell_s"] = df["dep_s"] - df["arr_s"]
    return df[["train_no", "role", "jambe", "station_id", "platform", "arrival", "departure", "dwell_s"]]
