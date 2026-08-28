"""Step 5 - test Dekker et al. (2022)'s exponential delay-loss claim.

Their diffusion model says a station sheds delay at a rate proportional to the
delay it currently carries: dD/dt = -B_i * D_i, i.e. delay loss is exponential
with a station-specific rate B_i.

Discretely, that predicts the change in a train's delay while it is at (or
running to) a point should be LINEAR in the delay it arrives with, with a
NEGATIVE slope. Slope magnitude estimates B_i per unit of the relevant interval.

Two tests per direction:
  dwell - (delay_dep - delay_arr) at a point, against delay_arr at that point
  run   - (delay_arr at k+1 - delay_dep at k), against delay_dep at k

A positive or flat slope falsifies the claim for that segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, load_traversals

ORDER = {"S2N": ["midi", "chapelle", "central", "congres", "nord"],
         "N2S": ["nord", "congres", "central", "chapelle", "midi"]}
MIN_N = 300


def ols(x: pd.Series, y: pd.Series) -> tuple[float, float, float, int]:
    m = ~(x.isna() | y.isna())
    x, y = x[m].to_numpy(float), y[m].to_numpy(float)
    if len(x) < MIN_N:
        return (np.nan,) * 3 + (len(x),)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, r, len(x)


def main() -> None:
    df = load_traversals()
    rows = []

    for d in ("S2N", "N2S"):
        sub = df[df["direction"] == d]
        for p in range(1, 6):
            name = ORDER[d][p - 1]
            # Congres and Chapelle are pass-through points: arr == dep for
            # 95-98% of trains by construction, so a dwell regression there is
            # definitionally zero and says nothing about Dekker's claim.
            stops = sub[sub[f"p{p}_stop_type"] == "="]
            passthrough = (sub[f"p{p}_delay_arr"] == sub[f"p{p}_delay_dep"]).mean()
            arr, dep = stops[f"p{p}_delay_arr"], stops[f"p{p}_delay_dep"]
            s, b, r, n = ols(arr, dep - arr)
            rows.append({"direction": d, "segment": "dwell", "pos": p, "point": name,
                         "slope": s, "intercept": b, "r": r, "n": n,
                         "passthrough_frac": round(float(passthrough), 3)})
        for p in range(1, 5):
            name = f"{ORDER[d][p-1]}->{ORDER[d][p]}"
            dep, nxt = sub[f"p{p}_delay_dep"], sub[f"p{p+1}_delay_arr"]
            s, b, r, n = ols(dep, nxt - dep)
            rows.append({"direction": d, "segment": "run", "pos": p, "point": name,
                         "slope": s, "intercept": b, "r": r, "n": n,
                         "passthrough_frac": float("nan")})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "step3_dekker.csv", index=False)

    print("Dekker exponential-decay test")
    print("  claim: slope < 0 (delay shed in proportion to delay carried)\n")
    for d in ("S2N", "N2S"):
        print(f"--- {d}")
        print(f"  {'segment':<8} {'point':<20} {'slope':>8} {'intercept':>10} {'r':>7} {'n':>7}  verdict")
        print("  (dwell rows restricted to trains that actually stop, stop_type '=')")
        for _, r in out[out["direction"] == d].iterrows():
            if np.isnan(r['slope']):
                v = "insufficient n"
            elif r["slope"] < -0.02:
                v = "consistent (sheds)"
            elif r["slope"] > 0.02:
                v = "CONTRADICTS (amplifies)"
            else:
                v = "flat (no effect)"
            print(f"  {r['segment']:<8} {r['point']:<20} {r['slope']:>8.3f} "
                  f"{r['intercept']:>10.1f} {r['r']:>7.3f} {r['n']:>7,}  {v}")
        print()

    ok = out[out["slope"] < -0.02]
    bad = out[out["slope"] > 0.02]
    print(f"segments consistent with exponential shedding: {len(ok)}/{len(out.dropna(subset=['slope']))}")
    print(f"segments contradicting (amplify delay)       : {len(bad)}")
    print(f"\n-> {OUT / 'step3_dekker.csv'}")


if __name__ == "__main__":
    main()
