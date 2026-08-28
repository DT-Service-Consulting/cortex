"""Shared helpers for Phase 0 analysis."""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAVERSALS = ROOT / "data" / "junction" / "traversals_202501.csv"
OUT = ROOT / "data" / "phase0"

# Positions p1..p5 are ordered along direction of travel.
POS = [1, 2, 3, 4, 5]
# Test on the last week; train on the rest. Temporal split, no shuffling.
TEST_DAYS = 25


def secs(t) -> float:
    """Seconds since midnight. Hours are not zero-padded in this feed."""
    if not isinstance(t, str) or not t.strip():
        return float("nan")
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def load_traversals() -> pd.DataFrame:
    df = pd.read_csv(TRAVERSALS, dtype=str)
    df["day"] = df["date"].str[:2].astype(int)
    df["entry_s"] = df["p1_real_dep"].map(secs)
    for c in ("entry_delay", "exit_delay", "delay_gained"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for p in POS:
        for f in ("delay_arr", "delay_dep"):
            df[f"p{p}_{f}"] = pd.to_numeric(df[f"p{p}_{f}"], errors="coerce")
    df["hour"] = (df["entry_s"] // 3600).astype("Int64")
    # service family from the relation label: IC 31 -> IC, L B1-2 -> L, P -> P
    df["service"] = df["relation"].str.strip().str.split().str[0].fillna("?")
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["day"] < TEST_DAYS].copy(), df[df["day"] >= TEST_DAYS].copy()


def scores(y_true, y_pred, threshold: float = 300.0) -> dict:
    """MAE/RMSE plus precision/recall for the 'late by >5 min' event."""
    import numpy as np
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true, y_pred = y_true[ok], y_pred[ok]
    err = y_true - y_pred
    at, ap = y_true > threshold, y_pred > threshold
    tp, fp, fn = (at & ap).sum(), (~at & ap).sum(), (at & ~ap).sum()
    return {
        "n": int(len(y_true)),
        "MAE": float(abs(err).mean()),
        "RMSE": float((err ** 2).mean() ** 0.5),
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
    }


def report(rows: dict[str, dict]) -> str:
    hdr = f"{'model':<26} {'n':>7} {'MAE':>8} {'RMSE':>9} {'prec':>7} {'recall':>7}"
    out = [hdr, "-" * len(hdr)]
    for name, s in rows.items():
        out.append(f"{name:<26} {s['n']:>7,} {s['MAE']:>8.1f} {s['RMSE']:>9.1f} "
                   f"{s['precision']:>7.3f} {s['recall']:>7.3f}")
    return "\n".join(out)
