"""Reusable delay-chain analysis.

The implementation is a production refactor of the methodology in
``Causality_Algorithm_Final.ipynb``, with hop delay added:

* station stay starts at the later of observed/simulated arrival and planned
  arrival;
* extra dwell is station stay minus planned dwell;
* extra arrival is observed arrival minus previous observed departure minus
  planned hop time (zero at the first stop of a trip);
* delay is the sum of extra dwell and extra arrival, each clipped at zero;
* calls with at least 30 seconds of delay form consecutive platform runs;
* runs of three or more calls are split when the platform gap is outside
  0--180 seconds or the adjacent delay change exceeds 180 seconds.

Typical use::

    pipeline = DelayChainPipeline(
        station_names={"215": "Bruxelles-Central"}
    )
    real = pipeline.analyze(real_df, planned_df, label="real")
    simulated_events = pipeline.prepare_simulated_events(sim_df, planned_df)
    simulated = pipeline.analyze(
        simulated_events, planned_df, label="simulated"
    )
    comparison = pipeline.compare({"real": real, "simulated": simulated})

All inputs are copied. The caller's DataFrames are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_timedelta64_dtype


DEFAULT_BRUSSELS_STATION_NAMES: dict[str, str] = {
    "215": "Bruxelles-Central",
    "216": "Bruxelles-Congrès",
    "217": "Bruxelles-Chapelle",
    "220": "Bruxelles-Midi",
    "221": "Bruxelles-Nord",
}


@dataclass(frozen=True)
class DelayChainConfig:
    """Thresholds used by the notebook's chain-refinement method."""

    positive_delay_threshold_seconds: float = 30.0
    minimum_platform_gap_seconds: float = 0.0
    maximum_platform_gap_seconds: float = 180.0
    maximum_delay_jump_seconds: float = 180.0
    minimum_chain_length_to_refine: int = 3
    elapsed_times_from_analysis_start: bool | None = None
    roll_over_midnight_departures: bool = False

    def __post_init__(self) -> None:
        if self.positive_delay_threshold_seconds < 0:
            raise ValueError("positive_delay_threshold_seconds must be non-negative")
        if self.minimum_platform_gap_seconds > self.maximum_platform_gap_seconds:
            raise ValueError(
                "minimum_platform_gap_seconds cannot exceed "
                "maximum_platform_gap_seconds"
            )
        if self.maximum_delay_jump_seconds < 0:
            raise ValueError("maximum_delay_jump_seconds must be non-negative")
        if self.minimum_chain_length_to_refine < 2:
            raise ValueError("minimum_chain_length_to_refine must be at least 2")


@dataclass(frozen=True)
class EventColumns:
    """Column mapping for a planned or event-side DataFrame."""

    trip_id: str = "trip_id"
    service_date: str = "service_date"
    sequence: str = "seq"
    direction: str = "direction"
    station_id: str = "station_id"
    platform: str = "platform"
    arrival: str = "arrival"
    departure: str = "departure"
    train_no: str | None = "train_no"
    source_train_no: str | None = "source_train_no"


@dataclass(frozen=True)
class SimulationColumns:
    """Column mapping for the simulation Parquet-style schema."""

    service_date: str = "datdep"
    service_date_format: str | None = "%d%b%Y"
    trip_id: str = "trip_id"
    source_train_no: str = "train_no"
    sequence: str = "seq"
    direction: str = "direction"
    station_id: str = "station_id"
    platform: str = "platform"
    arrival_seconds: str = "sim_arr_s"
    departure_seconds: str = "sim_dep_s"


@dataclass
class DelayChainResult:
    """Every reusable output produced for one dataset."""

    label: str
    events: pd.DataFrame
    event_universe: pd.DataFrame
    positive_chain_events: pd.DataFrame
    chain_events: pd.DataFrame
    chains: pd.DataFrame
    multi_event_chains: pd.DataFrame
    plausible_pairs: pd.DataFrame
    positive_chain_breakdown: pd.DataFrame
    refined_chain_breakdown: pd.DataFrame
    chain_length_distribution: pd.DataFrame
    propagation_profile: pd.DataFrame
    station_platform_breakdown: pd.DataFrame
    station_breakdown: pd.DataFrame
    station_platform_chain_length_counts: pd.DataFrame
    hour_profile: pd.DataFrame
    weekday_profile: pd.DataFrame
    date_profile: pd.DataFrame
    transfer_summary: pd.DataFrame
    summary: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return named tables for export or iterative comparison."""

        return {
            "events": self.events,
            "event_universe": self.event_universe,
            "positive_chain_events": self.positive_chain_events,
            "chain_events": self.chain_events,
            "chains": self.chains,
            "multi_event_chains": self.multi_event_chains,
            "plausible_pairs": self.plausible_pairs,
            "positive_chain_breakdown": self.positive_chain_breakdown,
            "refined_chain_breakdown": self.refined_chain_breakdown,
            "chain_length_distribution": self.chain_length_distribution,
            "propagation_profile": self.propagation_profile,
            "station_platform_breakdown": self.station_platform_breakdown,
            "station_breakdown": self.station_breakdown,
            "station_platform_chain_length_counts": (
                self.station_platform_chain_length_counts
            ),
            "hour_profile": self.hour_profile,
            "weekday_profile": self.weekday_profile,
            "date_profile": self.date_profile,
            "transfer_summary": self.transfer_summary,
            "summary": self.summary,
        }


@dataclass
class DelayChainComparison:
    """Side-by-side tables for two or more analysis results."""

    summary: pd.DataFrame
    station_platform: pd.DataFrame
    station: pd.DataFrame
    chain_lengths: pd.DataFrame
    propagation: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "summary": self.summary,
            "station_platform": self.station_platform,
            "station": self.station,
            "chain_lengths": self.chain_lengths,
            "propagation": self.propagation,
        }


class DelayChainPipeline:
    """Calculate and compare excess-dwell chains for arbitrary datasets."""

    _EVENT_REQUIRED = (
        "trip_id",
        "service_date",
        "seq",
        "direction",
        "station_id",
        "platform",
        "arrival",
        "departure",
    )

    def __init__(
        self,
        config: DelayChainConfig | None = None,
        station_names: Mapping[str | int, str] | None = None,
    ) -> None:
        self.config = config or DelayChainConfig()
        selected_names = (
            DEFAULT_BRUSSELS_STATION_NAMES
            if station_names is None
            else station_names
        )
        self.station_names = {
            str(station_id): name for station_id, name in selected_names.items()
        }

    def normalize_events(
        self,
        frame: pd.DataFrame,
        columns: EventColumns | None = None,
        *,
        arrival_time_unit: str | None = None,
        departure_time_unit: str | None = None,
    ) -> pd.DataFrame:
        """Return the canonical event schema used by :meth:`analyze`.

        Numeric arrival/departure values require explicit pandas time units,
        for example ``arrival_time_unit="s"``.
        """

        mapping = columns or EventColumns()
        rename_map = {
            mapping.trip_id: "trip_id",
            mapping.service_date: "service_date",
            mapping.sequence: "seq",
            mapping.direction: "direction",
            mapping.station_id: "station_id",
            mapping.platform: "platform",
            mapping.arrival: "arrival",
            mapping.departure: "departure",
        }
        if mapping.train_no is not None and mapping.train_no in frame.columns:
            rename_map[mapping.train_no] = "train_no"
        if (
            mapping.source_train_no is not None
            and mapping.source_train_no in frame.columns
        ):
            rename_map[mapping.source_train_no] = "source_train_no"

        missing_source_columns = sorted(set(rename_map).difference(frame.columns))
        if missing_source_columns:
            raise ValueError(
                "Input is missing mapped columns: "
                f"{missing_source_columns}"
            )

        normalized = frame.rename(columns=rename_map).copy()
        self._require_columns(normalized, self._EVENT_REQUIRED, "event data")
        normalized["service_date"] = pd.to_datetime(
            normalized["service_date"], errors="raise"
        ).dt.normalize()
        normalized["arrival"] = self._to_timedelta(
            normalized["arrival"], "arrival", arrival_time_unit
        )
        normalized["departure"] = self._to_timedelta(
            normalized["departure"], "departure", departure_time_unit
        )

        # The Final notebook clips negative station stays rather than inferring
        # midnight rollover. Other sources can opt into rollover explicitly.
        if self.config.roll_over_midnight_departures:
            overnight = normalized["departure"].lt(normalized["arrival"])
            normalized.loc[overnight, "departure"] += pd.Timedelta(days=1)

        normalized["trip_id"] = normalized["trip_id"].astype("string")
        normalized["station_id"] = normalized["station_id"].astype("string")
        normalized["platform"] = normalized["platform"].astype("string")
        normalized["direction"] = normalized["direction"].astype("string")
        if "train_no" not in normalized:
            normalized["train_no"] = normalized["trip_id"]
        if "source_train_no" not in normalized:
            normalized["source_train_no"] = normalized["train_no"]
        normalized["train_no"] = normalized["train_no"].astype("string")
        normalized["source_train_no"] = normalized[
            "source_train_no"
        ].astype("string")
        return normalized

    def prepare_simulated_events(
        self,
        simulation: pd.DataFrame,
        planned: pd.DataFrame,
        simulation_columns: SimulationColumns | None = None,
        planned_columns: EventColumns | None = None,
    ) -> pd.DataFrame:
        """Match Parquet-style simulation rows to their planned trip events.

        The operational key is service date, source train number, direction,
        sequence, station, and platform. Both sides must match one-to-one.
        """

        columns = simulation_columns or SimulationColumns()
        required = [
            columns.service_date,
            columns.trip_id,
            columns.source_train_no,
            columns.sequence,
            columns.direction,
            columns.station_id,
            columns.platform,
            columns.arrival_seconds,
            columns.departure_seconds,
        ]
        self._require_columns(simulation, required, "simulation data")

        planned_normalized = self.normalize_events(
            planned, columns=planned_columns
        )
        simulated = pd.DataFrame(
            {
                "service_date": pd.to_datetime(
                    simulation[columns.service_date],
                    format=columns.service_date_format,
                    errors="raise",
                ).dt.normalize(),
                "simulation_trip_id": simulation[columns.trip_id].astype("string"),
                "source_train_no": simulation[
                    columns.source_train_no
                ].astype("string"),
                "seq": simulation[columns.sequence].to_numpy(),
                "direction": simulation[columns.direction].astype("string"),
                "station_id": simulation[columns.station_id].astype("string"),
                "platform": simulation[columns.platform].astype("string"),
                "arrival": pd.to_timedelta(
                    simulation[columns.arrival_seconds], unit="s"
                ),
                "departure": pd.to_timedelta(
                    simulation[columns.departure_seconds], unit="s"
                ),
            }
        )
        if self.config.roll_over_midnight_departures:
            overnight = simulated["departure"].lt(simulated["arrival"])
            simulated.loc[overnight, "departure"] += pd.Timedelta(days=1)

        match_columns = [
            "service_date",
            "source_train_no",
            "direction",
            "seq",
            "station_id",
            "platform",
        ]
        planned_match = planned_normalized[
            match_columns + ["trip_id", "train_no"]
        ].rename(
            columns={
                "trip_id": "planned_trip_id",
                "train_no": "planned_train_no",
            }
        )
        self._reject_duplicates(simulated, match_columns, "simulation match key")
        self._reject_duplicates(planned_match, match_columns, "planned match key")

        matched = simulated.merge(
            planned_match,
            on=match_columns,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        unmatched = matched["_merge"].ne("both")
        if unmatched.any():
            sample = matched.loc[unmatched, match_columns].head(5).to_dict("records")
            raise ValueError(
                f"{unmatched.sum():,} simulation rows did not match planned "
                f"events. Sample keys: {sample}"
            )

        return pd.DataFrame(
            {
                "trip_id": matched["planned_trip_id"],
                "simulation_trip_id": matched["simulation_trip_id"],
                "train_no": matched["planned_train_no"],
                "source_train_no": matched["source_train_no"],
                "service_date": matched["service_date"],
                "seq": matched["seq"],
                "direction": matched["direction"],
                "station_id": matched["station_id"],
                "platform": matched["platform"],
                "arrival": matched["arrival"],
                "departure": matched["departure"],
            }
        )

    def analyze(
        self,
        events: pd.DataFrame,
        planned: pd.DataFrame,
        *,
        label: str = "dataset",
        event_columns: EventColumns | None = None,
        planned_columns: EventColumns | None = None,
        event_arrival_time_unit: str | None = None,
        event_departure_time_unit: str | None = None,
    ) -> DelayChainResult:
        """Run delay calculation, chain grouping, refinement, and breakdowns."""

        event_data = self.normalize_events(
            events,
            columns=event_columns,
            arrival_time_unit=event_arrival_time_unit,
            departure_time_unit=event_departure_time_unit,
        )
        planned_data = self.normalize_events(planned, columns=planned_columns)
        self._reject_duplicates(event_data, ["trip_id", "seq"], "event key")
        self._reject_duplicates(planned_data, ["trip_id", "seq"], "planned key")

        event_data["_original_order"] = range(len(event_data))
        event_data = event_data.sort_values(
            [
                "station_id",
                "platform",
                "service_date",
                "arrival",
                "trip_id",
                "seq",
            ],
            kind="stable",
        )
        event_data["chain"] = event_data.groupby(
            ["station_id", "platform"], sort=False, dropna=False
        ).cumcount()
        event_data = (
            event_data.sort_values("_original_order", kind="stable")
            .drop(columns="_original_order")
            .reset_index(drop=True)
        )

        planned_data["planned_station_stay_seconds"] = (
            planned_data["departure"] - planned_data["arrival"]
        ).dt.total_seconds()
        planned_lookup = planned_data[
            [
                "trip_id",
                "seq",
                "arrival",
                "departure",
                "planned_station_stay_seconds",
            ]
        ].rename(
            columns={
                "arrival": "planned_arrival",
                "departure": "planned_departure",
            }
        )
        event_data = event_data.merge(
            planned_lookup,
            on=["trip_id", "seq"],
            how="left",
            validate="one_to_one",
        )
        if event_data["planned_arrival"].isna().any():
            missing = event_data.loc[
                event_data["planned_arrival"].isna(), ["trip_id", "seq"]
            ].head(5)
            raise ValueError(
                "Some event rows do not match the planned data. Sample keys: "
                f"{missing.to_dict('records')}"
            )

        effective_arrival = event_data["arrival"].where(
            event_data["arrival"].ge(event_data["planned_arrival"]),
            event_data["planned_arrival"],
        )
        event_data["station_stay_seconds"] = (
            event_data["departure"] - effective_arrival
        ).dt.total_seconds().clip(lower=0)
        event_data["station_stay_delay_seconds"] = (
            event_data["station_stay_seconds"]
            - event_data["planned_station_stay_seconds"]
        )
        event_data["extra_dwell_seconds"] = event_data["station_stay_delay_seconds"]
        event_data = self._attach_extra_arrival(event_data)
        event_data["delay_seconds"] = (
            event_data["extra_dwell_seconds"].clip(lower=0)
            + event_data["extra_arrival_seconds"].clip(lower=0)
        )

        ordered = event_data.sort_values(
            ["station_id", "platform", "chain"], kind="stable"
        ).reset_index(drop=True)
        delay = ordered["delay_seconds"]
        is_positive = delay.ge(
            self.config.positive_delay_threshold_seconds
        ) & delay.notna()
        station_platform_change = ordered[["station_id", "platform"]].ne(
            ordered[["station_id", "platform"]].shift()
        ).any(axis=1)
        chain_number_break = ordered["chain"].ne(ordered["chain"].shift() + 1)
        condition_change = is_positive.ne(is_positive.shift(fill_value=False))
        ordered["_positive_run_id"] = (
            station_platform_change | chain_number_break | condition_change
        ).cumsum()

        platform_groups = ordered.groupby(
            ["station_id", "platform"], sort=False, dropna=False
        )
        ordered["previous_trip_id"] = platform_groups["trip_id"].shift()
        ordered["previous_chain_number"] = platform_groups["chain"].shift()
        ordered["previous_departure"] = platform_groups["departure"].shift()
        ordered["platform_gap_seconds"] = (
            ordered["arrival"] - ordered["previous_departure"]
        ).dt.total_seconds()
        ordered["platform_overlap_seconds"] = (
            -ordered["platform_gap_seconds"]
        ).clip(lower=0)

        positive_events = (
            ordered.loc[is_positive]
            .copy()
            .sort_values(
                ["station_id", "platform", "_positive_run_id", "chain"],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        positive_chain_breakdown = self._positive_chain_breakdown(positive_events)

        if positive_events.empty:
            chain_events, chains = self._empty_chain_outputs(positive_events)
        else:
            positive_events, chain_events, chains = self._refine_chains(
                positive_events, label
            )

        event_universe = self._add_event_timestamps(event_data)
        event_universe["station_platform"] = (
            event_universe["station_id"].astype(str)
            + " / "
            + event_universe["platform"].astype(str)
        )
        event_universe["is_positive_delay"] = event_universe[
            "delay_seconds"
        ].ge(self.config.positive_delay_threshold_seconds)

        multi_event_chains = chains.loc[
            chains["has_delayed_follower"]
        ].copy()
        multi_event_chains["passes_first_transition_rules"] = (
            multi_event_chains["second_platform_gap_seconds"].between(
                self.config.minimum_platform_gap_seconds,
                self.config.maximum_platform_gap_seconds,
                inclusive="both",
            )
            & multi_event_chains["second_minus_originator_seconds"].abs().le(
                self.config.maximum_delay_jump_seconds
            )
        )
        plausible_pairs = multi_event_chains.loc[
            multi_event_chains["passes_first_transition_rules"]
        ].copy()

        refined_chain_breakdown = self._refined_chain_breakdown(chains)
        chain_length_distribution = refined_chain_breakdown.rename(
            columns={
                "chain_length": "refined_chain_length",
                "number_of_refined_chains": "number_of_chains",
                "percent_of_refined_chains": "percent_of_chains",
            }
        )[
            ["refined_chain_length", "number_of_chains", "percent_of_chains"]
        ]
        propagation_profile = self._propagation_profile(chain_events)
        station_platform_breakdown = self._build_breakdown(
            event_universe, chains, plausible_pairs, ["station_name", "platform"]
        )
        station_breakdown = self._build_breakdown(
            event_universe, chains, plausible_pairs, ["station_name"]
        )
        chain_length_counts = self._build_exact_chain_length_counts(
            event_universe, chains
        )
        hour_profile = self._time_profile(
            event_universe, chains, "hour"
        )
        weekday_profile = self._time_profile(
            event_universe, chains, "weekday"
        )
        date_profile = self._time_profile(
            event_universe, chains, "calendar_date"
        )
        transfer_summary = self._transfer_summary(
            chains, multi_event_chains, plausible_pairs
        )
        summary = self._summary(
            label,
            event_universe,
            positive_chain_breakdown,
            chains,
            multi_event_chains,
            plausible_pairs,
        )

        return DelayChainResult(
            label=label,
            events=event_data,
            event_universe=event_universe,
            positive_chain_events=positive_events,
            chain_events=chain_events,
            chains=chains,
            multi_event_chains=multi_event_chains,
            plausible_pairs=plausible_pairs,
            positive_chain_breakdown=positive_chain_breakdown,
            refined_chain_breakdown=refined_chain_breakdown,
            chain_length_distribution=chain_length_distribution,
            propagation_profile=propagation_profile,
            station_platform_breakdown=station_platform_breakdown,
            station_breakdown=station_breakdown,
            station_platform_chain_length_counts=chain_length_counts,
            hour_profile=hour_profile,
            weekday_profile=weekday_profile,
            date_profile=date_profile,
            transfer_summary=transfer_summary,
            summary=summary,
        )

    def compare(
        self, results: Mapping[str, DelayChainResult]
    ) -> DelayChainComparison:
        """Build flat, export-friendly comparison tables for many datasets."""

        if len(results) < 2:
            raise ValueError("compare requires at least two named results")
        slugs = {name: self._slug(name) for name in results}
        if len(set(slugs.values())) != len(slugs):
            raise ValueError("Dataset names must remain distinct after slugification")

        summary_rows = []
        for name, result in results.items():
            row = result.summary.copy()
            row["dataset"] = name
            summary_rows.append(row)
        summary = pd.concat(summary_rows, ignore_index=True)

        station_platform = self._compare_tables(
            results,
            "station_platform_breakdown",
            ["station_name", "platform"],
            slugs,
        )
        station = self._compare_tables(
            results, "station_breakdown", ["station_name"], slugs
        )
        chain_lengths = self._compare_tables(
            results,
            "chain_length_distribution",
            ["refined_chain_length"],
            slugs,
        )
        propagation = self._compare_tables(
            results,
            "propagation_profile",
            ["position_in_refined_chain"],
            slugs,
        )
        return DelayChainComparison(
            summary=summary,
            station_platform=station_platform,
            station=station,
            chain_lengths=chain_lengths,
            propagation=propagation,
        )

    @staticmethod
    def _attach_extra_arrival(event_data: pd.DataFrame) -> pd.DataFrame:
        work = event_data.copy()
        work["_orig"] = range(len(work))
        work = work.sort_values(["trip_id", "seq"], kind="stable")
        grouped = work.groupby("trip_id", sort=False)
        work["previous_trip_departure"] = grouped["departure"].shift()
        work["previous_planned_departure"] = grouped["planned_departure"].shift()
        work["parkour_seconds"] = (
            work["planned_arrival"] - work["previous_planned_departure"]
        ).dt.total_seconds()
        work["extra_arrival_seconds"] = (
            (work["arrival"] - work["previous_trip_departure"]).dt.total_seconds()
            - work["parkour_seconds"]
        ).fillna(0)
        return (
            work.sort_values("_orig", kind="stable")
            .drop(columns="_orig")
            .reset_index(drop=True)
        )

    def _refine_chains(
        self, positive_events: pd.DataFrame, label: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        original_sizes = positive_events.groupby("_positive_run_id").size()
        positive_events["original_chain_length"] = positive_events[
            "_positive_run_id"
        ].map(original_sizes)
        groups = positive_events.groupby("_positive_run_id", sort=False)
        positive_events["previous_station_stay_delay_seconds"] = groups[
            "delay_seconds"
        ].shift()
        positive_events["delay_change_seconds"] = (
            positive_events["delay_seconds"]
            - positive_events["previous_station_stay_delay_seconds"]
        )
        positive_events["absolute_delay_change_seconds"] = positive_events[
            "delay_change_seconds"
        ].abs()

        is_start = groups.cumcount().eq(0)
        is_long = positive_events["original_chain_length"].ge(
            self.config.minimum_chain_length_to_refine
        )
        overlap = positive_events["platform_gap_seconds"].lt(
            self.config.minimum_platform_gap_seconds
        )
        excessive_gap = positive_events["platform_gap_seconds"].gt(
            self.config.maximum_platform_gap_seconds
        )
        excessive_jump = positive_events["absolute_delay_change_seconds"].gt(
            self.config.maximum_delay_jump_seconds
        )
        is_break = (
            ~is_start & is_long & (overlap | excessive_gap | excessive_jump)
        )
        positive_events["is_refinement_break"] = is_break

        reasons: list[str] = []
        for start, split, has_overlap, has_gap, has_jump in zip(
            is_start, is_break, overlap, excessive_gap, excessive_jump
        ):
            if start:
                reasons.append("original_chain_start")
            elif not split:
                reasons.append("continued")
            else:
                components = []
                if has_overlap:
                    components.append("platform_overlap")
                if has_gap:
                    components.append("large_platform_gap")
                if has_jump:
                    components.append("large_delay_jump")
                reasons.append("+".join(components))
        positive_events["refinement_break_reason"] = reasons
        positive_events["_starts_refined_chain"] = is_start | is_break
        positive_events["refined_segment_number"] = positive_events.groupby(
            "_positive_run_id", sort=False
        )["_starts_refined_chain"].cumsum().astype(int)
        positive_events["refined_chain_id"] = (
            positive_events["station_id"].astype(str)
            + ":"
            + positive_events["platform"].astype(str)
            + ":run-"
            + positive_events["_positive_run_id"].astype(str)
            + ":segment-"
            + positive_events["refined_segment_number"].astype(str)
        )

        chain_events = self._add_event_timestamps(positive_events).sort_values(
            ["refined_chain_id", "chain"], kind="stable"
        )
        groups = chain_events.groupby("refined_chain_id", sort=False)
        chain_events["position_in_refined_chain"] = groups.cumcount() + 1
        chain_events["refined_chain_length"] = groups["trip_id"].transform("size")
        chain_events["originator_delay_seconds"] = groups[
            "delay_seconds"
        ].transform("first")
        chain_events["previous_delay_in_refined_chain_seconds"] = groups[
            "delay_seconds"
        ].shift()
        chain_events["change_from_previous_seconds"] = (
            chain_events["delay_seconds"]
            - chain_events["previous_delay_in_refined_chain_seconds"]
        )
        chain_events["change_from_originator_seconds"] = (
            chain_events["delay_seconds"]
            - chain_events["originator_delay_seconds"]
        )
        chain_events["fraction_of_originator_delay"] = (
            chain_events["delay_seconds"]
            / chain_events["originator_delay_seconds"]
        )

        groups = chain_events.groupby("refined_chain_id", sort=False)
        first = groups.first()
        last = groups.last()
        aggregate = groups.agg(
            refined_chain_length=("trip_id", "size"),
            mean_chain_delay_seconds=("delay_seconds", "mean"),
            maximum_chain_delay_seconds=("delay_seconds", "max"),
            minimum_chain_delay_seconds=("delay_seconds", "min"),
            chain_start_timestamp=("event_timestamp", "min"),
            chain_end_timestamp=("event_timestamp", "max"),
        )
        first_two = (
            chain_events.loc[
                chain_events["position_in_refined_chain"].le(2),
                [
                    "refined_chain_id",
                    "position_in_refined_chain",
                    "delay_seconds",
                    "platform_gap_seconds",
                    "planned_station_stay_seconds",
                    "seq",
                ],
            ]
            .set_index(["refined_chain_id", "position_in_refined_chain"])
            .unstack("position_in_refined_chain")
        )
        first_two.columns = [
            f"{column}_{int(position)}" for column, position in first_two.columns
        ]
        for column in (
            "delay_seconds_1",
            "delay_seconds_2",
            "platform_gap_seconds_1",
            "platform_gap_seconds_2",
            "planned_station_stay_seconds_1",
            "planned_station_stay_seconds_2",
            "seq_1",
            "seq_2",
        ):
            if column not in first_two:
                first_two[column] = float("nan")

        chains = (
            aggregate.join(first_two)
            .assign(
                station_id=first["station_id"],
                platform=first["platform"],
                station_platform=(
                    first["station_id"].astype(str)
                    + " / "
                    + first["platform"].astype(str)
                ),
                direction=first["direction"],
                originator_trip_id=first["trip_id"],
                last_trip_id=last["trip_id"],
                originator_source_train_no=first["source_train_no"],
                originator_start_reason=first["refinement_break_reason"],
                last_delay_seconds=last["delay_seconds"],
            )
            .rename(
                columns={
                    "delay_seconds_1": "originator_delay_seconds",
                    "delay_seconds_2": "second_delay_seconds",
                    "platform_gap_seconds_1": "originator_platform_gap_seconds",
                    "platform_gap_seconds_2": "second_platform_gap_seconds",
                    "planned_station_stay_seconds_1": (
                        "originator_planned_dwell_seconds"
                    ),
                    "planned_station_stay_seconds_2": "second_planned_dwell_seconds",
                    "seq_1": "originator_seq",
                    "seq_2": "second_seq",
                }
            )
            .reset_index()
        )
        chains["dataset"] = label
        chains["follower_count"] = chains["refined_chain_length"] - 1
        chains["has_delayed_follower"] = chains["follower_count"].gt(0)
        chains["second_minus_originator_seconds"] = (
            chains["second_delay_seconds"] - chains["originator_delay_seconds"]
        )
        chains["second_to_originator_ratio"] = (
            chains["second_delay_seconds"] / chains["originator_delay_seconds"]
        )
        chains["last_minus_originator_seconds"] = (
            chains["last_delay_seconds"] - chains["originator_delay_seconds"]
        )
        chains["chain_span_seconds"] = (
            chains["chain_end_timestamp"] - chains["chain_start_timestamp"]
        ).dt.total_seconds()
        chains["calendar_date"] = chains["chain_start_timestamp"].dt.date
        chains["hour"] = chains["chain_start_timestamp"].dt.hour
        chains["weekday"] = chains["chain_start_timestamp"].dt.day_name()
        return positive_events, chain_events, chains

    def _empty_chain_outputs(
        self, positive_events: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        event_columns = {
            "original_chain_length": "int64",
            "previous_station_stay_delay_seconds": "float64",
            "delay_change_seconds": "float64",
            "absolute_delay_change_seconds": "float64",
            "is_refinement_break": "bool",
            "refinement_break_reason": "string",
            "_starts_refined_chain": "bool",
            "refined_segment_number": "int64",
            "refined_chain_id": "string",
            "event_timestamp": "datetime64[ns]",
            "calendar_date": "object",
            "hour": "int64",
            "weekday": "string",
            "position_in_refined_chain": "int64",
            "refined_chain_length": "int64",
            "originator_delay_seconds": "float64",
            "previous_delay_in_refined_chain_seconds": "float64",
            "change_from_previous_seconds": "float64",
            "change_from_originator_seconds": "float64",
            "fraction_of_originator_delay": "float64",
        }
        chain_events = positive_events.copy()
        for column, dtype in event_columns.items():
            chain_events[column] = pd.Series(index=chain_events.index, dtype=dtype)

        chain_columns = [
            "refined_chain_id",
            "refined_chain_length",
            "mean_chain_delay_seconds",
            "maximum_chain_delay_seconds",
            "minimum_chain_delay_seconds",
            "chain_start_timestamp",
            "chain_end_timestamp",
            "originator_delay_seconds",
            "second_delay_seconds",
            "originator_platform_gap_seconds",
            "second_platform_gap_seconds",
            "originator_planned_dwell_seconds",
            "second_planned_dwell_seconds",
            "originator_seq",
            "second_seq",
            "station_id",
            "platform",
            "station_platform",
            "direction",
            "originator_trip_id",
            "last_trip_id",
            "originator_source_train_no",
            "originator_start_reason",
            "last_delay_seconds",
            "dataset",
            "follower_count",
            "has_delayed_follower",
            "second_minus_originator_seconds",
            "second_to_originator_ratio",
            "last_minus_originator_seconds",
            "chain_span_seconds",
            "calendar_date",
            "hour",
            "weekday",
        ]
        chains = pd.DataFrame(columns=chain_columns)
        chains["has_delayed_follower"] = chains[
            "has_delayed_follower"
        ].astype(bool)
        return chain_events, chains

    def _add_event_timestamps(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        mode = self.config.elapsed_times_from_analysis_start
        if mode is None:
            mode = bool(output["arrival"].max() >= pd.Timedelta(days=1))
        if mode:
            start = output["service_date"].min().normalize()
            output["event_timestamp"] = start + output["arrival"]
        else:
            output["event_timestamp"] = output["service_date"] + output["arrival"]
        output["calendar_date"] = output["event_timestamp"].dt.date
        output["hour"] = output["event_timestamp"].dt.hour
        output["weekday"] = output["event_timestamp"].dt.day_name()
        return output

    def _positive_chain_breakdown(
        self, positive_events: pd.DataFrame
    ) -> pd.DataFrame:
        columns = [
            "chain_length",
            "number_of_positive_chains",
            "total_events_in_these_chains",
            "percent_of_positive_chains",
            "percent_of_positive_events",
        ]
        if positive_events.empty:
            return pd.DataFrame(columns=columns)
        lengths = positive_events.groupby("_positive_run_id").size()
        output = (
            lengths.value_counts()
            .sort_index()
            .rename_axis("chain_length")
            .reset_index(name="number_of_positive_chains")
        )
        output["total_events_in_these_chains"] = (
            output["chain_length"] * output["number_of_positive_chains"]
        )
        output["percent_of_positive_chains"] = (
            100 * output["number_of_positive_chains"] / len(lengths)
        )
        output["percent_of_positive_events"] = (
            100
            * output["total_events_in_these_chains"]
            / len(positive_events)
        )
        return output[columns]

    @staticmethod
    def _refined_chain_breakdown(chains: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "chain_length",
            "number_of_refined_chains",
            "total_events_in_these_chains",
            "percent_of_refined_chains",
            "percent_of_refined_events",
        ]
        if chains.empty:
            return pd.DataFrame(columns=columns)
        output = (
            chains["refined_chain_length"]
            .value_counts()
            .sort_index()
            .rename_axis("chain_length")
            .reset_index(name="number_of_refined_chains")
        )
        output["total_events_in_these_chains"] = (
            output["chain_length"] * output["number_of_refined_chains"]
        )
        output["percent_of_refined_chains"] = (
            100 * output["number_of_refined_chains"] / len(chains)
        )
        total_events = output["total_events_in_these_chains"].sum()
        output["percent_of_refined_events"] = (
            100 * output["total_events_in_these_chains"] / total_events
        )
        return output[columns]

    @staticmethod
    def _propagation_profile(chain_events: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "position_in_refined_chain",
            "event_count",
            "mean_delay_seconds",
            "median_delay_seconds",
            "lower_quartile_seconds",
            "upper_quartile_seconds",
        ]
        if chain_events.empty:
            return pd.DataFrame(columns=columns)
        return (
            chain_events.groupby("position_in_refined_chain", observed=True)[
                "delay_seconds"
            ]
            .agg(
                event_count="size",
                mean_delay_seconds="mean",
                median_delay_seconds="median",
                lower_quartile_seconds=lambda values: values.quantile(0.25),
                upper_quartile_seconds=lambda values: values.quantile(0.75),
            )
            .reset_index()
        )

    def _build_breakdown(
        self,
        events: pd.DataFrame,
        chains: pd.DataFrame,
        plausible_pairs: pd.DataFrame,
        group_columns: list[str],
    ) -> pd.DataFrame:
        named_events = self._add_station_names(events)
        named_chains = self._add_station_names(chains)
        named_plausible = self._add_station_names(plausible_pairs)
        calls = (
            named_events.groupby(group_columns, observed=True, dropna=False)
            .agg(
                station_calls=("trip_id", "size"),
                positive_delay_calls=("is_positive_delay", "sum"),
                positive_delay_rate=("is_positive_delay", "mean"),
            )
            .reset_index()
        )
        chain_metrics = [
            "refined_chains",
            "chains_length_1",
            "chains_length_2",
            "chains_length_3_plus",
            "mean_chain_length",
            "median_chain_length",
            "maximum_chain_length",
        ]
        if named_chains.empty:
            chain_table = pd.DataFrame(columns=group_columns + chain_metrics)
            pair_table = pd.DataFrame(
                columns=group_columns
                + [
                    "second_train_pairs",
                    "mean_second_delay_seconds",
                    "median_second_delay_seconds",
                    "second_delay_p25_seconds",
                    "second_delay_p75_seconds",
                    "median_second_minus_originator_seconds",
                    "median_second_to_originator_ratio",
                ]
            )
        else:
            chain_table = (
                named_chains.groupby(group_columns, observed=True, dropna=False)
                .agg(
                    refined_chains=("refined_chain_id", "size"),
                    chains_length_1=(
                        "refined_chain_length", lambda values: values.eq(1).sum()
                    ),
                    chains_length_2=(
                        "refined_chain_length", lambda values: values.eq(2).sum()
                    ),
                    chains_length_3_plus=(
                        "refined_chain_length", lambda values: values.ge(3).sum()
                    ),
                    mean_chain_length=("refined_chain_length", "mean"),
                    median_chain_length=("refined_chain_length", "median"),
                    maximum_chain_length=("refined_chain_length", "max"),
                )
                .reset_index()
            )
            followers = named_chains.loc[named_chains["has_delayed_follower"]]
            pair_table = (
                followers.groupby(group_columns, observed=True, dropna=False)
                .agg(
                    second_train_pairs=("refined_chain_id", "size"),
                    mean_second_delay_seconds=("second_delay_seconds", "mean"),
                    median_second_delay_seconds=("second_delay_seconds", "median"),
                    second_delay_p25_seconds=(
                        "second_delay_seconds", lambda values: values.quantile(0.25)
                    ),
                    second_delay_p75_seconds=(
                        "second_delay_seconds", lambda values: values.quantile(0.75)
                    ),
                    median_second_minus_originator_seconds=(
                        "second_minus_originator_seconds", "median"
                    ),
                    median_second_to_originator_ratio=(
                        "second_to_originator_ratio", "median"
                    ),
                )
                .reset_index()
            )

        plausible_metrics = [
            "plausible_second_train_pairs",
            "plausible_median_second_delay_seconds",
            "plausible_median_second_minus_originator_seconds",
        ]
        if named_plausible.empty:
            plausible_table = pd.DataFrame(
                columns=group_columns + plausible_metrics
            )
        else:
            plausible_table = (
                named_plausible.groupby(
                    group_columns, observed=True, dropna=False
                )
                .agg(
                    plausible_second_train_pairs=("refined_chain_id", "size"),
                    plausible_median_second_delay_seconds=(
                        "second_delay_seconds", "median"
                    ),
                    plausible_median_second_minus_originator_seconds=(
                        "second_minus_originator_seconds", "median"
                    ),
                )
                .reset_index()
            )

        output = (
            calls.merge(chain_table, on=group_columns, how="left")
            .merge(pair_table, on=group_columns, how="left")
            .merge(plausible_table, on=group_columns, how="left")
        )
        count_columns = [
            "refined_chains",
            "chains_length_1",
            "chains_length_2",
            "chains_length_3_plus",
            "second_train_pairs",
            "plausible_second_train_pairs",
        ]
        for column in count_columns:
            output[column] = (
                pd.to_numeric(output[column], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
        output["chains_with_second_train_percent"] = self._safe_percent(
            output["second_train_pairs"], output["refined_chains"]
        )
        output["plausible_share_of_second_pairs_percent"] = self._safe_percent(
            output["plausible_second_train_pairs"], output["second_train_pairs"]
        )
        output["multi_train_chains_per_1000_calls"] = (
            1000 * output["second_train_pairs"] / output["station_calls"]
        )
        sort_columns = list(group_columns)
        if "platform" in sort_columns:
            output["_platform_sort"] = pd.to_numeric(
                output["platform"], errors="coerce"
            )
            sort_columns = [
                "_platform_sort" if column == "platform" else column
                for column in sort_columns
            ]
        output = output.sort_values(sort_columns, kind="stable")
        return output.drop(columns="_platform_sort", errors="ignore").reset_index(
            drop=True
        )

    def _build_exact_chain_length_counts(
        self, events: pd.DataFrame, chains: pd.DataFrame
    ) -> pd.DataFrame:
        named_events = self._add_station_names(events)
        calls = (
            named_events.groupby(
                ["station_name", "platform"], observed=True, dropna=False
            )
            .size()
            .rename("station_calls")
            .to_frame()
        )
        if chains.empty:
            output = calls.assign(chains_length_1=0).reset_index()
        else:
            named_chains = self._add_station_names(chains)
            counts = (
                named_chains.groupby(
                    ["station_name", "platform", "refined_chain_length"],
                    observed=True,
                    dropna=False,
                )
                .size()
                .unstack("refined_chain_length", fill_value=0)
            )
            maximum = int(named_chains["refined_chain_length"].max())
            counts = counts.reindex(columns=range(1, maximum + 1), fill_value=0)
            counts.columns = [
                f"chains_length_{length}" for length in counts.columns
            ]
            output = calls.join(counts, how="left").fillna(0).reset_index()
            length_columns = [
                column
                for column in output
                if column.startswith("chains_length_")
            ]
            output[length_columns] = output[length_columns].astype(int)
        output["_platform_sort"] = pd.to_numeric(
            output["platform"], errors="coerce"
        )
        return (
            output.sort_values(
                ["station_name", "_platform_sort"], kind="stable"
            )
            .drop(columns="_platform_sort")
            .reset_index(drop=True)
        )

    @staticmethod
    def _time_profile(
        events: pd.DataFrame, chains: pd.DataFrame, time_column: str
    ) -> pd.DataFrame:
        event_profile = (
            events.groupby(time_column, observed=True)
            .agg(
                event_count=("trip_id", "size"),
                positive_event_count=("is_positive_delay", "sum"),
                positive_event_rate=("is_positive_delay", "mean"),
            )
            .reset_index()
        )
        if chains.empty:
            event_profile["refined_chain_starts"] = 0
            event_profile["multi_train_chain_starts"] = 0
            event_profile["median_second_minus_originator_seconds"] = float("nan")
        else:
            chain_profile = (
                chains.groupby(time_column, observed=True)
                .agg(
                    refined_chain_starts=("refined_chain_id", "size"),
                    multi_train_chain_starts=("has_delayed_follower", "sum"),
                    median_second_minus_originator_seconds=(
                        "second_minus_originator_seconds", "median"
                    ),
                )
                .reset_index()
            )
            event_profile = event_profile.merge(
                chain_profile, on=time_column, how="left"
            )
            event_profile[["refined_chain_starts", "multi_train_chain_starts"]] = (
                event_profile[
                    ["refined_chain_starts", "multi_train_chain_starts"]
                ]
                .fillna(0)
                .astype(int)
            )
        event_profile["multi_train_chains_per_1000_calls"] = (
            1000
            * event_profile["multi_train_chain_starts"]
            / event_profile["event_count"]
        )
        return event_profile

    def _transfer_summary(
        self,
        chains: pd.DataFrame,
        pairs: pd.DataFrame,
        plausible_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        slope = self._slope(pairs)
        plausible_slope = self._slope(plausible_pairs)
        correlation = (
            pairs["originator_delay_seconds"].corr(pairs["second_delay_seconds"])
            if len(pairs) >= 2
            else float("nan")
        )
        metrics = {
            "all_refined_chains": len(chains),
            "chains_with_follower": len(pairs),
            "share_with_follower": self._scalar_ratio(len(pairs), len(chains)),
            "median_originator_delay_seconds": pairs[
                "originator_delay_seconds"
            ].median(),
            "median_second_delay_seconds": pairs["second_delay_seconds"].median(),
            "median_second_minus_originator_seconds": pairs[
                "second_minus_originator_seconds"
            ].median(),
            "mean_second_minus_originator_seconds": pairs[
                "second_minus_originator_seconds"
            ].mean(),
            "median_second_to_originator_ratio": pairs[
                "second_to_originator_ratio"
            ].median(),
            "ols_second_seconds_per_originator_second": slope,
            "ols_second_seconds_per_originator_minute": 60 * slope,
            "originator_second_pearson_correlation": correlation,
            "plausible_pairs": len(plausible_pairs),
            "plausible_share": self._scalar_ratio(
                len(plausible_pairs), len(pairs)
            ),
            "plausible_median_second_minus_originator_seconds": plausible_pairs[
                "second_minus_originator_seconds"
            ].median(),
            "plausible_ols_second_seconds_per_originator_minute": (
                60 * plausible_slope
            ),
        }
        return pd.DataFrame(
            {"metric": list(metrics), "value": list(metrics.values())}
        )

    @staticmethod
    def _summary(
        label: str,
        events: pd.DataFrame,
        positive_breakdown: pd.DataFrame,
        chains: pd.DataFrame,
        pairs: pd.DataFrame,
        plausible_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        positive_calls = int(events["is_positive_delay"].sum())
        original_runs = (
            int(positive_breakdown["number_of_positive_chains"].sum())
            if not positive_breakdown.empty
            else 0
        )
        return pd.DataFrame(
            [
                {
                    "dataset": label,
                    "station_calls": len(events),
                    "positive_delay_calls": positive_calls,
                    "positive_delay_rate": (
                        positive_calls / len(events) if len(events) else float("nan")
                    ),
                    "mean_excess_dwell_all_calls_seconds": events[
                        "station_stay_delay_seconds"
                    ].mean(),
                    "median_excess_dwell_all_calls_seconds": events[
                        "station_stay_delay_seconds"
                    ].median(),
                    "original_positive_runs": original_runs,
                    "refined_chains": len(chains),
                    "multi_train_chains": len(pairs),
                    "multi_train_chains_per_1000_calls": (
                        1000 * len(pairs) / len(events) if len(events) else float("nan")
                    ),
                    "mean_refined_chain_length": chains[
                        "refined_chain_length"
                    ].mean(),
                    "maximum_refined_chain_length": chains[
                        "refined_chain_length"
                    ].max(),
                    "median_originator_delay_seconds": pairs[
                        "originator_delay_seconds"
                    ].median(),
                    "median_second_delay_seconds": pairs[
                        "second_delay_seconds"
                    ].median(),
                    "median_second_minus_originator_seconds": pairs[
                        "second_minus_originator_seconds"
                    ].median(),
                    "plausible_second_train_pairs": len(plausible_pairs),
                }
            ]
        )

    def _add_station_names(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        if "station_id" not in output:
            output["station_id"] = pd.Series(dtype="string")
        codes = output["station_id"].astype("string")
        output["station_name"] = codes.map(self.station_names).fillna(
            "Station " + codes
        )
        return output

    @staticmethod
    def _compare_tables(
        results: Mapping[str, DelayChainResult],
        attribute: str,
        keys: list[str],
        slugs: Mapping[str, str],
    ) -> pd.DataFrame:
        output: pd.DataFrame | None = None
        for name, result in results.items():
            table = getattr(result, attribute).copy()
            metrics = [column for column in table if column not in keys]
            table = table.rename(
                columns={column: f"{slugs[name]}_{column}" for column in metrics}
            )
            output = (
                table
                if output is None
                else output.merge(table, on=keys, how="outer")
            )
        assert output is not None
        return output.sort_values(keys, kind="stable").reset_index(drop=True)

    @staticmethod
    def _to_timedelta(
        values: pd.Series, name: str, unit: str | None
    ) -> pd.Series:
        if is_timedelta64_dtype(values.dtype):
            return values.copy()
        if is_numeric_dtype(values.dtype) and unit is None:
            raise ValueError(
                f"Numeric {name} values require an explicit {name}_time_unit"
            )
        return pd.to_timedelta(values, unit=unit, errors="raise")

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame, columns: list[str] | tuple[str, ...], context: str
    ) -> None:
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"{context} is missing required columns: {missing}")

    @staticmethod
    def _reject_duplicates(
        frame: pd.DataFrame, columns: list[str], context: str
    ) -> None:
        duplicate = frame.duplicated(columns, keep=False)
        if duplicate.any():
            sample = frame.loc[duplicate, columns].head(5).to_dict("records")
            raise ValueError(
                f"{context} must be unique; found {duplicate.sum():,} duplicate "
                f"rows. Sample keys: {sample}"
            )

    @staticmethod
    def _safe_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        result = pd.Series(0.0, index=numerator.index)
        valid = denominator.gt(0)
        result.loc[valid] = 100 * numerator.loc[valid] / denominator.loc[valid]
        return result

    @staticmethod
    def _scalar_ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else float("nan")

    @staticmethod
    def _slope(pairs: pd.DataFrame) -> float:
        if len(pairs) < 2:
            return float("nan")
        originator = pairs["originator_delay_seconds"]
        variance = originator.var()
        if pd.isna(variance) or variance == 0:
            return float("nan")
        return originator.cov(pairs["second_delay_seconds"]) / variance

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
        if not slug:
            raise ValueError(f"Dataset name cannot be slugified: {value!r}")
        return slug


__all__ = [
    "DEFAULT_BRUSSELS_STATION_NAMES",
    "DelayChainComparison",
    "DelayChainConfig",
    "DelayChainPipeline",
    "DelayChainResult",
    "EventColumns",
    "SimulationColumns",
]
