from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx
import pandas as pd

from .delay_chain_pipeline import DelayChainPipeline
from .simulation.build_hard import PASSAGES_FILE, STATIONS, build_scenario
from .simulation.causal_ground import CORRIDOR, clock, jambe_spec, score_edges
from .simulation.ConstrainedRouter import ConstrainedRouter
from .simulation.month import compare, netconvert, read_passages, read_stop_events, run_sumo, write_configs

HOP_KM = (1.129, 0.700, 0.900, 1.091)
N_JAMBES = 6
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
        n_chains: int = 6,
        n_stations: int = 5,
        inject_station_no: int = 2,
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
        service_date: str = "2024-01-01",
        infra_dir: str | Path | None = None,
        out_dir: str | Path | None = None,
    ):
        root = Path(__file__).resolve().parents[1]
        self.n_trains = n_trains
        self.n_chains = max(1, min(n_chains, 6, n_trains))
        self.n_stations = n_stations
        self.inject_station_no = inject_station_no
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
        self.compare_df = pd.DataFrame()
        self.pipeline_planned = pd.DataFrame()
        self.pipeline_observed = pd.DataFrame()
        self.detected_edges = pd.DataFrame()
        self.detection_score: dict | None = None
        self.delay_chain_result = None
        self.service_date = pd.Timestamp(service_date).normalize()
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
        extra_by_first = {}
        dwell_by_first = {}
        for chain in self.chains:
            extra = max(
                self.first_dwell_time - self.dwell_time,
                (len(chain) - 1) * self.headway_time + 1,
            )
            extra_by_first[chain[0]] = extra
            dwell_by_first[chain[0]] = self.dwell_time + extra
        planning_abs = self.planning.copy()
        first_by_platform = {p: chain[0] for p, chain in enumerate(self.chains, start=1)}
        for i, r in planning_abs.iterrows():
            first = first_by_platform[r["PLATFORM_NO"]]
            if r["TRAIN_NO"] != first:
                continue
            extra = extra_by_first[first]
            if r["STATION_NO"] == self.inject_station_no:
                planning_abs.at[i, "DEP_TIME"] = r["ARR_TIME"] + dwell_by_first[first]
            elif r["STATION_NO"] > self.inject_station_no:
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
            jambe = _jambe(int(r["PLATFORM_NO"]) - 1)
            direction, plats = jambe_spec(jambe)
            seq = int(r["STATION_NO"]) - 1
            pno = jambe + 1
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
                "source_train_no": str(r["TRAIN_NO"]),
                "service_date": self.service_date,
                "seq": seq,
                "direction": direction,
                "station_id": sid,
                "platform": plat,
                "arrival": clock(arr_s),
                "departure": clock(dep_s),
                "jambe": jambe,
                "station_no": int(r["STATION_NO"]),
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
        run_sumo(day_dir, end=self.sim_end, hold_next_stop=False)
        stop_file = day_dir / "stop_events.xml"
        if not stop_file.exists():
            raise RuntimeError("sumo n'a pas produit stop_events.xml")
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
        keep = [
            "trip_id", "train_no", "chain_id", "chain_pos", "role", "seq", "direction",
            "station", "station_id", "platform",
            "plan_arr", "plan_dep", "sim_arr", "sim_dep",
            "sim_arr_s", "sim_dep_s", "dwell_s", "delay_arr_s", "delay_dep_s",
        ]
        events = events[[c for c in keep if c in events.columns]]
        events = events.sort_values(["chain_id", "chain_pos", "seq"]).reset_index(drop=True)
        self.compare_df = cmp
        self.events = events
        self._save(cmp, scenario)
        return events

    def conflict_station_ids(self) -> list[int]:
        seq = self.inject_station_no - 1
        ids = []
        for cid in range(len(self.chains)):
            direction, _ = jambe_spec(_jambe(cid))
            sid = int(CORRIDOR[direction][seq])
            if sid not in ids:
                ids.append(sid)
        return ids

    def planned_for_pipeline(self) -> pd.DataFrame:
        if self.tt_ref.empty:
            self.generate()
        self.pipeline_planned = self._pipeline_frame(
            self.tt_ref, self.tt_ref["arrival"], self.tt_ref["departure"]
        )
        return self.pipeline_planned

    def observed_for_pipeline(self, source: str = "simulation") -> pd.DataFrame:
        if source not in {"simulation", "timetable"}:
            raise ValueError("source must be 'simulation' or 'timetable'")
        if source == "timetable":
            if self.tt_abs.empty:
                self.generate()
            self.pipeline_observed = self._pipeline_frame(
                self.tt_abs, self.tt_abs["arrival"], self.tt_abs["departure"]
            )
            return self.pipeline_observed
        if self.compare_df.empty:
            self.simulate()
        cmp = self.compare_df
        self.pipeline_observed = self._pipeline_frame(
            cmp,
            pd.to_timedelta(cmp["sim_arr_s"], unit="s"),
            pd.to_timedelta(cmp["sim_dep_s"], unit="s"),
        )
        return self.pipeline_observed

    def run_delay_chain(
        self,
        source: str = "simulation",
        pipeline: DelayChainPipeline | None = None,
    ) -> dict:
        planned = self.planned_for_pipeline()
        observed = self.observed_for_pipeline(source)
        pipeline = pipeline or DelayChainPipeline()
        result = pipeline.analyze(observed, planned, label=source)
        conflict = {str(s) for s in self.conflict_station_ids()}
        edges = self._detected_edges(result)
        at_conflict = (
            edges[edges["station_id"].astype(str).isin(conflict)].copy()
            if len(edges)
            else edges
        )
        truth = []
        if not self.gt_edges.empty:
            truth = [
                (str(a), str(b))
                for a, b in self.gt_edges[["cause", "effect"]].itertuples(index=False)
            ]
        pred = []
        if len(at_conflict):
            pred = [
                (str(a), str(b))
                for a, b in at_conflict[["cause", "effect"]].itertuples(index=False)
            ]
        score = score_edges(truth, pred)
        report = self._detection_report(result)
        self.delay_chain_result = result
        self.detected_edges = at_conflict
        self.detection_score = {
            **score,
            "source": source,
            "conflict_station_ids": self.conflict_station_ids(),
            "inject_station_no": self.inject_station_no,
        }
        self._save_detection(result, edges, report)
        return {
            "result": result,
            "score": self.detection_score,
            "edges": at_conflict,
            "edges_all_stations": edges,
            "report": report,
            "planned": planned,
            "observed": observed,
        }

    def _pipeline_frame(self, base: pd.DataFrame, arrival, departure) -> pd.DataFrame:
        return pd.DataFrame({
            "trip_id": base["trip_id"].astype("string").to_numpy(),
            "train_no": base["train_no"].astype("string").to_numpy(),
            "source_train_no": base["train_no"].astype("string").to_numpy(),
            "service_date": self.service_date,
            "seq": base["seq"].to_numpy(),
            "direction": base["direction"].astype("string").to_numpy(),
            "station_id": base["station_id"].astype("string").to_numpy(),
            "platform": base["platform"].astype("string").to_numpy(),
            "arrival": pd.Series(arrival).to_numpy(),
            "departure": pd.Series(departure).to_numpy(),
        })

    @staticmethod
    def _detected_edges(result) -> pd.DataFrame:
        events = result.chain_events
        if events.empty:
            return pd.DataFrame(columns=[
                "cause", "effect", "station_id", "platform", "direction", "refined_chain_id",
            ])
        multi = events.loc[events["refined_chain_length"].ge(2)].copy()
        rows = []
        for chain_id, group in multi.groupby("refined_chain_id", sort=False):
            group = group.sort_values("position_in_refined_chain", kind="stable")
            trains = group["train_no"].astype(str).tolist()
            station_id = str(group["station_id"].iloc[0])
            platform = str(group["platform"].iloc[0])
            direction = str(group["direction"].iloc[0])
            for cause, effect in zip(trains, trains[1:]):
                rows.append({
                    "cause": cause,
                    "effect": effect,
                    "station_id": station_id,
                    "platform": platform,
                    "direction": direction,
                    "refined_chain_id": chain_id,
                })
        return pd.DataFrame(rows)

    def _detection_report(self, result) -> pd.DataFrame:
        conflict = {str(s) for s in self.conflict_station_ids()}
        universe = result.event_universe
        universe = universe[universe["station_id"].astype(str).isin(conflict)].copy()
        chains = result.chains
        if not chains.empty:
            chains = chains[chains["station_id"].astype(str).isin(conflict)].copy()
        rows = []
        for cid, chain in enumerate(self.chains):
            direction, _ = jambe_spec(_jambe(cid))
            sid = str(CORRIDOR[direction][self.inject_station_no - 1])
            loc = self.tt_ref[
                (self.tt_ref["train_no"] == chain[0])
                & (self.tt_ref["station_id"].astype(str) == sid)
            ]
            platform = str(loc["platform"].iloc[0]) if len(loc) else ""
            at = universe[
                (universe["station_id"].astype(str) == sid)
                & (universe["platform"].astype(str) == platform)
            ]
            delayed = (
                at.loc[at["is_positive_delay"], "train_no"].astype(str).tolist()
                if len(at)
                else []
            )
            found = []
            if not chains.empty:
                hit = chains[
                    (chains["station_id"].astype(str) == sid)
                    & (chains["platform"].astype(str) == platform)
                    & chains["refined_chain_length"].ge(2)
                ]
                for _, row in hit.iterrows():
                    members = result.chain_events
                    members = members[members["refined_chain_id"] == row["refined_chain_id"]]
                    members = members.sort_values("position_in_refined_chain", kind="stable")
                    found.append(members["train_no"].astype(str).tolist())
            rows.append({
                "chain_id": cid,
                "direction": direction,
                "station_id": sid,
                "platform": platform,
                "true_chain": chain,
                "positive_delay_trains": delayed,
                "detected_chains": found,
                "true_edge_count": max(0, len(chain) - 1),
                "detected_multi_train": bool(found),
            })
        return pd.DataFrame(rows)

    def _save_detection(self, result, edges: pd.DataFrame, report: pd.DataFrame) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_planned.to_csv(self.out_dir / "pipeline_planned.csv", index=False)
        self.pipeline_observed.to_csv(self.out_dir / "pipeline_observed.csv", index=False)
        if not self.planning.empty:
            self.planning.to_csv(self.out_dir / "planning_ref.csv", index=False)
            self.planning_abs.to_csv(self.out_dir / "planning_abs.csv", index=False)
            self.tt_ref.to_csv(self.out_dir / "timetable_ref.csv", index=False)
            self.tt_abs.to_csv(self.out_dir / "timetable_abs.csv", index=False)
        edges.to_csv(self.out_dir / "detected_edges.csv", index=False)
        report.to_csv(self.out_dir / "detection_report.csv", index=False)
        result.summary.to_csv(self.out_dir / "delay_chain_summary.csv", index=False)
        result.station_platform_breakdown.to_csv(
            self.out_dir / "delay_chain_station_platform.csv", index=False
        )
        score = self.detection_score or {}
        serializable = {
            key: ([list(item) for item in value] if key in {"tp", "fp", "fn"} else value)
            for key, value in score.items()
        }
        (self.out_dir / "detection_score.json").write_text(json.dumps(serializable, indent=2))
        if not self.gt_edges.empty:
            self.gt_edges.to_csv(self.out_dir / "ground_truth_edges.csv", index=False)
        (self.out_dir / "chain_groups.json").write_text(json.dumps(self.chains, indent=2))
        print("score", serializable)
        print("->", self.out_dir)

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


def _jambe(chain_id: int) -> int:
    return int(chain_id) % N_JAMBES


def _fmt_clock(t) -> str | None:
    if pd.isna(t):
        return None
    t = int(round(float(t)))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"
