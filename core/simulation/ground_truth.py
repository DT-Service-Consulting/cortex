"""Generate synthetic causal-delay ground truth with SUMO."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import networkx as nx
import pandas as pd

from .build_hard import PASSAGES_FILE, STATIONS, build_scenario
from .causal_ground import CORRIDOR, clock, jambe_spec
from .ConstrainedRouter import ConstrainedRouter
from .month import compare, netconvert, read_passages, read_stop_events, write_configs

HOP_KM = (1.129, 0.700, 0.900, 1.091)
INFRA_FILES = (
    "all_tracks_bcong.csv",
    "station_track_assigned.csv",
    "switches.csv",
    "forbidden_transitions.csv",
    "constrained_digraph.pkl",
    "oriented_edges.pkl",
)


class GroundTruthGenerator:
    def __init__(
        self,
        n_trains: int = 10,
        n_stations: int = 5,
        dwell_time: float = 1,
        first_dwell_time: float = 10,
        headway_time: float = 2,
        chain_gap_time: float = 2 * 60,
        speed: float = 40,
        effect_strength_min: float = 0.1,
        effect_strength_max: float = 1.0,
        seed: int | None = None,
        t0_clock: int = 8 * 3600,
        sim_end: int = 24 * 3600,
        infra_dir: str | Path | None = None,
        out_dir: str | Path | None = None,
    ):
        root = Path(__file__).resolve().parents[2]
        self.n_trains = n_trains
        self.n_chains = 6
        self.n_stations = n_stations
        self.dwell_time = dwell_time
        self.first_dwell_time = first_dwell_time
        self.headway_time = headway_time
        self.chain_gap_time = chain_gap_time
        self.speed = speed
        self.effect_strength_min = effect_strength_min
        self.effect_strength_max = effect_strength_max
        self.seed = seed
        self.t0_clock = t0_clock
        self.sim_end = sim_end
        self.infra_dir = Path(infra_dir) if infra_dir else root / ".personal" / "simulation"
        self.out_dir = Path(out_dir) if out_dir else self.infra_dir / "causal_gt"
        self.travel_time = pd.DataFrame([
            {"STATION_NO": i, "NEXT_STATION_NO": i + 1, "TRAVEL_TIME": km / speed * 60}
            for i, km in enumerate(HOP_KM, start=1)
        ])
        self.graph: nx.DiGraph | None = None
        self.chains: list[list[str]] = []
        self.planning = pd.DataFrame()
        self.planning_abs = pd.DataFrame()
        self.gt_edges = pd.DataFrame()
        self.tt_ref = pd.DataFrame()
        self.tt_abs = pd.DataFrame()
        self.events = pd.DataFrame()
        self._router = None
        self._tracks = None
        self._assigned = None
        self._switches = None
        self._forbidden = None

    def generate_chains(self, seed: int | None = None) -> tuple[nx.DiGraph, list[list[str]]]:
        if seed is None:
            seed = self.seed
        if seed is not None:
            random.seed(seed)
        n_chains = max(1, min(self.n_chains, self.n_trains))
        train_ids = [f"TRAIN_{i}" for i in range(self.n_trains)]
        random.shuffle(train_ids)
        splits, remaining = [], self.n_trains
        for i in range(n_chains - 1):
            size = random.randint(1, remaining - (n_chains - i - 1))
            splits.append(size)
            remaining -= size
        splits.append(remaining)
        chains, idx = [], 0
        for size in splits:
            chains.append(train_ids[idx:idx + size])
            idx += size
        G = nx.DiGraph()
        for chain in chains:
            for train_no in chain:
                G.add_node(train_no, param=random.uniform(self.effect_strength_min, self.effect_strength_max))
            for a, b in zip(chain, chain[1:]):
                G.add_edge(
                    a, b,
                    effect_strength=round(random.uniform(self.effect_strength_min, self.effect_strength_max), 2),
                )
        self.graph, self.chains = G, chains
        return G, chains

    def build_planning(self) -> pd.DataFrame:
        if not self.chains:
            self.generate_chains()
        run = dict(zip(self.travel_time["STATION_NO"], self.travel_time["TRAVEL_TIME"]))
        rows = []
        chain_t0 = 0.0
        for p, chain in enumerate(self.chains, start=1):
            prev_dep = {}
            last_dep = chain_t0
            for train in chain:
                t = chain_t0 if not prev_dep else prev_dep[1] + self.headway_time
                for s in range(1, self.n_stations + 1):
                    if s in prev_dep:
                        t = max(t, prev_dep[s] + self.headway_time)
                    arr, dep = t, t + self.dwell_time
                    rows.append({
                        "TRAIN_NO": train,
                        "STATION_NO": s,
                        "ARR_TIME": arr,
                        "DEP_TIME": dep,
                        "PLATFORM_NO": p,
                    })
                    prev_dep[s] = dep
                    last_dep = max(last_dep, dep)
                    if s < self.n_stations:
                        t = dep + float(run[s])
            chain_t0 = last_dep + self.chain_gap_time
        self.planning = pd.DataFrame(rows)
        return self.planning

    def build_planning_abs(self) -> pd.DataFrame:
        if self.planning.empty:
            self.build_planning()
        extra = self.first_dwell_time - self.dwell_time
        planning_abs = self.planning.copy()
        first_by_platform = {p: chain[0] for p, chain in enumerate(self.chains, start=1)}
        for i, r in planning_abs.iterrows():
            if r["TRAIN_NO"] != first_by_platform[r["PLATFORM_NO"]]:
                continue
            if r["STATION_NO"] == 2:
                planning_abs.at[i, "DEP_TIME"] = r["ARR_TIME"] + self.first_dwell_time
            elif r["STATION_NO"] > 2:
                planning_abs.at[i, "ARR_TIME"] = r["ARR_TIME"] + extra
                planning_abs.at[i, "DEP_TIME"] = r["DEP_TIME"] + extra
        self.planning_abs = planning_abs
        self.gt_edges = pd.DataFrame(
            [{"cause": a, "effect": b} for chain in self.chains for a, b in zip(chain, chain[1:])]
        )
        return planning_abs

    def generate(self, seed: int | None = None) -> "GroundTruthGenerator":
        self.generate_chains(seed=seed)
        self.build_planning()
        self.build_planning_abs()
        self.tt_ref = self.to_timetable(self.planning)
        self.tt_abs = self.to_timetable(self.planning_abs)
        return self

    def to_timetable(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            jambe = int(r["PLATFORM_NO"]) - 1
            direction, plats = jambe_spec(jambe)
            seq = int(r["STATION_NO"]) - 1
            pno = int(r["PLATFORM_NO"])
            sid = int(CORRIDOR[direction][seq])
            if sid == 220:
                plat = pno + 7
            elif sid == 221:
                plat = pno + 3
            else:
                plat = int(plats[sid])
            arr_s = self.t0_clock + int(round(float(r["ARR_TIME"]) * 60))
            dep_s = self.t0_clock + int(round(float(r["DEP_TIME"]) * 60))
            rows.append({
                "trip_id": f"GT:{r['TRAIN_NO']}:{direction}",
                "train_no": str(r["TRAIN_NO"]),
                "seq": seq,
                "direction": direction,
                "station_id": sid,
                "platform": plat,
                "arrival": clock(arr_s),
                "departure": clock(dep_s),
                "jambe": jambe,
            })
        return pd.DataFrame(rows)

    def simulate(self) -> pd.DataFrame:
        if self.planning_abs.empty:
            self.generate()
        self._load_infra()
        day_dir = self.out_dir / "full"
        day_dir.mkdir(parents=True, exist_ok=True)
        scenario = build_scenario(
            tracks=self._tracks,
            assigned=self._assigned,
            switches=self._switches,
            router=self._router,
            forbidden_df=self._forbidden,
            timetable=self.tt_abs,
            out_dir=day_dir,
            t0=0,
        )
        write_configs(day_dir, end=self.sim_end)
        nc = netconvert(day_dir)
        if nc.returncode != 0:
            raise RuntimeError(nc.stderr[-800:])
        run = subprocess.run(
            ["sumo", "-c", str(day_dir / "ns.batch.sumocfg")],
            capture_output=True,
            text=True,
        )
        stop_file = day_dir / "stop_events.xml"
        if not stop_file.exists():
            raise RuntimeError((run.stderr or run.stdout)[-800:])
        stop_events = read_stop_events(stop_file)
        crossings = day_dir / PASSAGES_FILE
        passages = read_passages(crossings) if crossings.exists() else pd.DataFrame()
        cmp = compare(self.tt_ref, stop_events, passages)
        cmp["station"] = cmp["station_id"].map(STATIONS)
        meta = self._train_meta()
        events = cmp.copy()
        events["chain_id"] = events["train_no"].map(lambda t: meta[t]["chain_id"])
        events["chain_pos"] = events["train_no"].map(lambda t: meta[t]["chain_pos"])
        events["role"] = events["train_no"].map(lambda t: meta[t]["role"])
        events["sim_arr"] = events["sim_arr_s"].map(_fmt_clock)
        events["sim_dep"] = events["sim_dep_s"].map(_fmt_clock)
        events["dwell_s"] = (events["sim_dep_s"] - events["sim_arr_s"]).round(1)
        events["delay_arr_s"] = events["delay_arr_s"].round(1)
        events["delay_dep_s"] = events["delay_dep_s"].round(1)
        events = events.rename(columns={"arrival": "plan_arr", "departure": "plan_dep"})
        events = events[[
            "train_no", "chain_id", "chain_pos", "role", "seq", "direction",
            "station", "station_id", "platform",
            "plan_arr", "plan_dep", "sim_arr", "sim_dep",
            "dwell_s", "delay_arr_s", "delay_dep_s",
        ]].sort_values(["chain_id", "chain_pos", "seq"]).reset_index(drop=True)
        self.events = events
        self._save(cmp, scenario)
        return events

    def _train_meta(self) -> dict[str, dict]:
        meta = {}
        for cid, chain in enumerate(self.chains):
            for pos, train in enumerate(chain):
                if len(chain) == 1:
                    role = "isolated"
                elif pos == 0:
                    role = "cause"
                else:
                    role = "effect"
                meta[train] = {"chain_id": cid, "chain_pos": pos, "role": role}
        return meta

    def _load_infra(self) -> None:
        if self._router is not None:
            return
        need = [self.infra_dir / name for name in INFRA_FILES]
        if not all(p.exists() for p in need):
            raise FileNotFoundError("infra cache manquante, lancer research/simulate.ipynb d'abord")
        self._tracks = pd.read_csv(self.infra_dir / "all_tracks_bcong.csv", sep=";")
        self._assigned = pd.read_csv(self.infra_dir / "station_track_assigned.csv")
        self._switches = pd.read_csv(self.infra_dir / "switches.csv", sep=";")
        self._forbidden = pd.read_csv(self.infra_dir / "forbidden_transitions.csv")
        self._router = ConstrainedRouter.fromFiles(
            str(self.infra_dir / "constrained_digraph.pkl"),
            str(self.infra_dir / "oriented_edges.pkl"),
        )

    def _save(self, cmp: pd.DataFrame, scenario: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.gt_edges.to_csv(self.out_dir / "ground_truth_edges.csv", index=False)
        self.planning.to_csv(self.out_dir / "planning_ref.csv", index=False)
        self.planning_abs.to_csv(self.out_dir / "planning_abs.csv", index=False)
        self.tt_ref.to_csv(self.out_dir / "timetable_ref.csv", index=False)
        self.tt_abs.to_csv(self.out_dir / "timetable_abs.csv", index=False)
        self.events.to_csv(self.out_dir / "stop_events.csv", index=False)
        cmp.to_csv(self.out_dir / "compare.csv", index=False)
        (self.out_dir / "chain_groups.json").write_text(json.dumps(self.chains, indent=2))
        print("failed", scenario["failed"])
        print("->", self.out_dir)


def _fmt_clock(t) -> str | None:
    if pd.isna(t):
        return None
    t = int(round(float(t)))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"
