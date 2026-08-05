from __future__ import annotations

import ast
from typing import Any

import pandas as pd


def _point(path_pt: list) -> tuple[float, float]:
    return float(path_pt[0]), float(path_pt[1])


def rebuild_switches_from_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Rebuild switches.csv from track path endpoints (ID, X=lat, Y=lon).

    Same idea as the original SUMO prep pipeline: every Departure/Arrival switch
    gets coordinates from Path[0] / Path[-1]. After a track split, re-run this
    so new intermediate switches exist.
    """
    coords: dict[int, tuple[float, float]] = {}
    for _, r in tracks.iterrows():
        path = ast.literal_eval(r["Path"]) if isinstance(r["Path"], str) else r["Path"]
        if not isinstance(path, list) or len(path) < 2:
            continue
        dep, arr = int(r["Departure_switch"]), int(r["Arrival_switch"])
        coords.setdefault(dep, _point(path[0]))
        coords.setdefault(arr, _point(path[-1]))

    rows = [{"ID": sid, "X": xy[0], "Y": xy[1]} for sid, xy in sorted(coords.items())]
    return pd.DataFrame(rows, columns=["ID", "X", "Y"])


def ensure_switches_cover_tracks(
    tracks: pd.DataFrame,
    switches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return switches covering every track endpoint; fill missing from Path."""
    rebuilt = rebuild_switches_from_tracks(tracks)
    if switches is None or switches.empty:
        return rebuilt

    sw = switches.copy()
    sw["ID"] = sw["ID"].astype(int)
    have = set(sw["ID"].tolist())
    need = set(rebuilt["ID"].tolist())
    missing = need - have
    if not missing:
        return sw.sort_values("ID").reset_index(drop=True)

    add = rebuilt[rebuilt["ID"].isin(missing)]
    out = pd.concat([sw[["ID", "X", "Y"]], add], ignore_index=True)
    return out.sort_values("ID").reset_index(drop=True)
