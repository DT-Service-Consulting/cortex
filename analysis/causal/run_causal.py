"""Causal analysis of knock-on delay at the Brussels North-South junction.

Question: when a train enters the junction close behind another, does the leader's
state *cause* the follower to lose time, or do the two merely share upstream causes?

Phase 0 established the association (r ~ +0.11, stable across 12 months, unmoved by
weather / hour / month). Association is not the claim we want. Three designs:

  A. Constraint-based discovery (PC and FCI) over a tiered variable set, with the
     temporal order supplied as background knowledge. FCI matters because it admits
     latent confounders instead of inventing direct edges for them.

  B. A negative control. If the coupling is a physical track-occupancy mechanism it
     must WEAKEN as trains move apart, and vanish once they are far enough apart to
     be operationally independent (Li et al. 2024 put that at ~20 min). If instead
     it is driven by shared upstream disruption, it should persist at ALL headways,
     because a disruption hits near and far trains alike. This is the sharpest test
     available on observational data, because the two stories predict opposite
     things.

  C. An instrumental variable. Scheduled headway comes from a timetable fixed months
     ahead, so it cannot be caused by today's disruption, but it strongly drives
     actual headway (first-stage F ~ 2.7e5). Conditioning on hour/month/track/
     direction, the residual variation in scheduled headway is plausibly exogenous.

Two statistical hazards are handled explicitly:
  - n is ~313k, so every independence test rejects. Significance is uninformative;
    we use subsampling with stability selection and report EFFECT SIZES.
  - Delays are heavy-tailed, so raw partial correlations are unreliable. All
    continuous variables are rank-transformed to normal scores first.

Usage: python run_causal.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
from step7_full_year import load_all, secs  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data" / "causal"
RNG = np.random.default_rng(0)

# Tiers encode what cannot cause what. Nothing may point back to an earlier tier.
TIERS = {
    0: ["sched_hw", "hour", "month_i", "temp", "rain", "snow"],  # fixed before the day
    1: ["leader_entry"],                                          # leader enters first
    2: ["act_hw", "foll_entry"],                                  # follower enters
    3: ["foll_gain"],                                             # outcome, last
}
VARS = [v for t in sorted(TIERS) for v in TIERS[t]]


def normal_scores(a: np.ndarray) -> np.ndarray:
    """Rank-transform to normal scores - tames the heavy tail, keeps monotone order."""
    r = stats.rankdata(a)
    return stats.norm.ppf(r / (len(r) + 1.0))


def build() -> pd.DataFrame:
    df = load_all()
    df["planned_entry_s"] = df["p1_planned_dep"].map(secs)
    df = df.dropna(subset=["planned_entry_s", "entry_s", "entry_delay",
                           "exit_delay", "delay_gained"])
    df = df.sort_values(["date", "direction", "tunnel_track", "entry_s"])
    g = df.groupby(["date", "direction", "tunnel_track"], sort=False)
    df["sched_hw"] = df["planned_entry_s"] - g["planned_entry_s"].shift(1)
    df["act_hw"] = df["entry_s"] - g["entry_s"].shift(1)
    df["leader_entry"] = g["entry_delay"].shift(1)
    df = df.rename(columns={"entry_delay": "foll_entry", "delay_gained": "foll_gain"})
    df = df.dropna(subset=["sched_hw", "act_hw", "leader_entry"])
    df = df[df["act_hw"].between(1, 7200) & df["sched_hw"].between(1, 7200)]
    df["month_i"] = df["month"].str[-2:].astype(int)

    w = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "phase0" /
                    "weather_2025.csv", dtype={"ptcar": str})
    w = w[w["ptcar"] == "215"][["wdate", "hour", "temperature_2m", "rain", "snowfall"]]
    w.columns = ["date", "hour", "temp", "rain", "snow"]
    df["hour"] = df["hour"].astype(int)
    df = df.merge(w, on=["date", "hour"], how="left").dropna(subset=["temp"])
    return df


def partial_corr(x, y, Z) -> float:
    """Partial correlation of x and y given the columns of Z."""
    Z = np.column_stack([Z, np.ones(len(Z))])
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(x - Z @ bx, y - Z @ by)[0, 1])


# ---------------------------------------------------------------- design A
def discovery(D: pd.DataFrame, n_sub: int = 4000, reps: int = 40) -> None:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.graph.GraphNode import GraphNode

    X = D[VARS].to_numpy(float)
    idx = {v: i for i, v in enumerate(VARS)}
    tier_of = {v: t for t, vs in TIERS.items() for v in vs}

    nodes = [GraphNode(v) for v in VARS]
    bk = BackgroundKnowledge()
    for a in VARS:
        for b in VARS:
            if a != b and tier_of[a] > tier_of[b]:
                bk.add_forbidden_by_node(nodes[idx[a]], nodes[idx[b]])

    print(f"\nvariables ({len(VARS)}): {VARS}")
    print(f"background knowledge: {sum(tier_of[a] > tier_of[b] for a in VARS for b in VARS if a != b)} "
          f"arrows forbidden by temporal tier")
    print(f"stability selection: {reps} subsamples of n={n_sub:,} "
          f"(full n={len(D):,} would make every test significant)")

    for name, run in (("PC", "pc"), ("FCI", "fci")):
        counts = np.zeros((len(VARS), len(VARS)))
        ok = 0
        for _ in range(reps):
            s = X[RNG.choice(len(X), n_sub, replace=False)]
            try:
                # causal-learn narrates its orientation steps to stdout; mute it
                with contextlib.redirect_stdout(io.StringIO()):
                    if run == "pc":
                        G = pc(s, 0.01, fisherz, background_knowledge=bk,
                               node_names=VARS, show_progress=False).G.graph
                    else:
                        G = fci(s, fisherz, 0.01, background_knowledge=bk,
                                node_names=VARS, verbose=False)[0].graph
            except Exception:
                continue
            ok += 1
            counts += (G != 0).astype(float)
        if not ok:
            print(f"\n{name}: all runs failed")
            continue
        freq = counts / ok
        print(f"\n--- {name}: edges present in >=50% of {ok} subsamples ---")
        seen = set()
        rows = []
        for i, a in enumerate(VARS):
            for j, b in enumerate(VARS):
                if i >= j or freq[i, j] < 0.5 or (i, j) in seen:
                    continue
                seen.add((i, j))
                rows.append((freq[i, j], a, b))
        for f, a, b in sorted(rows, reverse=True):
            star = "  <<<" if {a, b} == {"leader_entry", "foll_gain"} else ""
            print(f"   {f:5.0%}  {a:<14} -- {b}{star}")
        key = freq[idx["leader_entry"], idx["foll_gain"]]
        print(f"   leader_entry -- foll_gain retained in {key:.0%} of subsamples")


# ---------------------------------------------------------------- design B
def negative_control(D: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("DESIGN B - negative control by headway")
    print("  mechanism predicts: effect decays to zero as trains separate")
    print("  confounding predicts: effect roughly constant at all headways")
    print("=" * 78)
    ctrl = ["hour", "month_i", "temp", "rain", "snow"]
    bands = [(0, 180, "<3 min"), (180, 300, "3-5"), (300, 600, "5-10"),
             (600, 1200, "10-20"), (1200, 2400, "20-40"), (2400, 7200, "40-120")]
    print(f"\n  {'headway':<10} {'n':>8} {'raw r':>9} {'partial r':>11}")
    for lo, hi, lab in bands:
        b = D[D["act_hw"].between(lo, hi, inclusive="left")]
        if len(b) < 500:
            continue
        x = normal_scores(b["leader_entry"].values)
        y = normal_scores(b["foll_gain"].values)
        Z = b[ctrl].to_numpy(float)
        print(f"  {lab:<10} {len(b):>8,} {np.corrcoef(x, y)[0, 1]:>9.3f} "
              f"{partial_corr(x, y, Z):>11.3f}")


# ---------------------------------------------------------------- design C
def iv(D: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("DESIGN C - scheduled headway as an instrument for actual headway")
    print("=" * 78)
    d = D[D["act_hw"] < 2400].copy()
    ctrl = pd.get_dummies(d[["hour", "month_i", "tunnel_track", "direction"]],
                          columns=["hour", "month_i", "tunnel_track", "direction"],
                          drop_first=True).astype(float).to_numpy()
    ones = np.ones((len(d), 1))
    C = np.column_stack([ctrl, ones])
    z = d["sched_hw"].to_numpy(float)
    x = d["act_hw"].to_numpy(float)
    y = d["foll_gain"].to_numpy(float)

    def resid(v):
        return v - C @ np.linalg.lstsq(C, v, rcond=None)[0]

    zr, xr, yr = resid(z), resid(x), resid(y)
    # OLS
    ols = float(np.dot(xr, yr) / np.dot(xr, xr))
    # 2SLS: project x on z, then y on x-hat
    first = float(np.dot(zr, xr) / np.dot(zr, zr))
    xhat = zr * first
    tsls = float(np.dot(xhat, yr) / np.dot(xhat, xhat))
    r2 = 1 - (xr - xhat).var() / xr.var()
    F = r2 / (1 - r2) * (len(d) - 2)
    print(f"  n = {len(d):,}   controls: hour, month, track, direction")
    print(f"  first stage: d(actual)/d(scheduled) = {first:+.3f}   partial R2 = {r2:.3f}   F = {F:,.0f}")
    print(f"\n  OLS   d(delay gained)/d(headway) = {ols * 60:+.2f} s per extra minute of headway")
    print(f"  2SLS  d(delay gained)/d(headway) = {tsls * 60:+.2f} s per extra minute of headway")
    print("\n  Negative = more headway means less delay gained, i.e. congestion relief.")
    print("  OLS is biased: actual headway is itself shortened by disruption, which")
    print("  also raises delay. 2SLS uses only the timetable-driven part of headway.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D = build()
    print(f"pairs for causal analysis: {len(D):,}")
    D.to_csv(OUT / "pairs_causal_2025.csv", index=False)

    Dn = D.copy()
    for v in ["sched_hw", "act_hw", "leader_entry", "foll_entry", "foll_gain",
              "temp", "rain", "snow"]:
        Dn[v] = normal_scores(D[v].to_numpy(float))

    print("\n" + "=" * 78)
    print("DESIGN A - constraint-based discovery with temporal background knowledge")
    print("=" * 78)
    discovery(Dn)
    negative_control(D)
    iv(D)


if __name__ == "__main__":
    main()
