"""Step 7 - fetch weather and test whether it improves the Phase 0 model.

Replicates RIDE's recipe (arXiv 2606.05070, appendix "Weather"): hourly data from
the Open-Meteo Historical Weather API, queried per operational point using its
latitude/longitude, timezone Europe/Brussels, with the same six variables:
temperature_2m, rain, snowfall, relative_humidity_2m, wind_speed_10m, weather_code.

Weather is attached at the ENTRY point (p1) and the entry hour - the prediction
moment, matching RIDE's "weather variables for each operational point at
prediction time".

Usage: python step5_weather.py
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from common import OUT, ROOT, report, scores, split
from step2_baselines import FEATURES_BASE, FEATURES_INTER, TARGET, fit_predict

API = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["temperature_2m", "rain", "snowfall", "relative_humidity_2m",
        "wind_speed_10m", "weather_code"]
JUNCTION = {"215", "216", "217", "220", "221"}
START, END = "2024-12-31", "2025-02-01"   # one day of margin either side
CACHE = OUT / "weather_202501.csv"

ENTRY_PTCAR = {"S2N": "220", "N2S": "221"}


def fetch() -> pd.DataFrame:
    pts = pd.read_csv(ROOT / "data" / "reference" / "operational_points.csv",
                      dtype={"ptcarid": str})
    pts = pts[pts["ptcarid"].isin(JUNCTION)]

    frames = []
    for _, p in pts.iterrows():
        q = urllib.parse.urlencode({
            "latitude": p["lat"], "longitude": p["lon"],
            "start_date": START, "end_date": END,
            "hourly": ",".join(VARS), "timezone": "Europe/Brussels",
        })
        req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "cortex/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        h = pd.DataFrame(d["hourly"])
        h["ptcar"] = p["ptcarid"]
        frames.append(h)
        print(f"  {p['ptcarid']} {p['longnamefrench'][:22]:<22} {len(h):,} hourly rows")
        time.sleep(1.0)

    w = pd.concat(frames, ignore_index=True)
    w["time"] = pd.to_datetime(w["time"])
    w["wdate"] = w["time"].dt.strftime("%d%b%Y").str.upper()
    w["hour"] = w["time"].dt.hour
    return w


def profile(w: pd.DataFrame) -> None:
    print("\nJanuary 2025 at Bruxelles-Central (215):")
    jan = w[(w["ptcar"] == "215") & (w["wdate"].str.endswith("JAN2025"))]
    print(jan[VARS].describe().loc[["mean", "min", "max"]].round(2).to_string())
    print(f"\n  hours with rain > 0     : {(jan['rain'] > 0).sum():,} / {len(jan):,}")
    print(f"  hours with snowfall > 0 : {(jan['snowfall'] > 0).sum():,}")
    print(f"  hours below 0 C         : {(jan['temperature_2m'] < 0).sum():,}")

    piv = w[w["wdate"].str.endswith("JAN2025")].pivot_table(
        index=["wdate", "hour"], columns="ptcar", values="temperature_2m")
    spread = piv.max(axis=1) - piv.min(axis=1)
    print(f"\nmax-min temperature across the 5 points, same hour: "
          f"mean {spread.mean():.3f} C, max {spread.max():.2f} C")
    print("  -> Open-Meteo's grid is coarser than the 3.5 km junction: all five points")
    print("     return identical values, so per-point querying buys nothing here.")


def join_weather(pairs: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["entry_ptcar"] = pairs["direction"].map(ENTRY_PTCAR)
    cols = ["ptcar", "wdate", "hour"] + VARS
    out = pairs.merge(w[cols], how="left",
                      left_on=["entry_ptcar", "date", "hour"],
                      right_on=["ptcar", "wdate", "hour"])
    return out.drop(columns=["ptcar", "wdate"])


def evaluate(w: pd.DataFrame) -> None:
    pairs = pd.read_csv(OUT / "pairs_202501.csv")
    pairs["day"] = pairs["date"].str[:2].astype(int)
    pairs = (pairs.dropna(subset=[TARGET, "follower_entry_delay", "headway_s"])
                  .rename(columns={"follower_entry_delay": "entry_delay"}))
    df = join_weather(pairs, w).dropna(subset=VARS)
    tr, te = split(df)
    te = te.reset_index(drop=True)
    base = FEATURES_BASE + FEATURES_INTER

    pb, _ = fit_predict(tr, te, base)
    pw, _ = fit_predict(tr, te, base + VARS)
    res = {"persistence": scores(te[TARGET], te["entry_delay"]),
           "GBDT (+ headway/leader)": scores(te[TARGET], pb),
           "GBDT (+ weather)": scores(te[TARGET], pw)}
    print("\n=== does weather improve prediction? ===")
    print(report(res))
    d = res["GBDT (+ headway/leader)"]["MAE"] - res["GBDT (+ weather)"]["MAE"]
    print(f"weather block is worth {d:+.2f}s MAE "
          f"({d / res['GBDT (+ headway/leader)']['MAE'] * 100:+.2f}%)")

    bad = (te["rain"] > 0.5) | (te["snowfall"] > 0) | (te["temperature_2m"] < 0)
    print(f"\nbad-weather subgroup (rain>0.5mm | snow | sub-zero), n={bad.sum():,}:")
    print(report({"persistence": scores(te[TARGET][bad], te["entry_delay"][bad]),
                  "GBDT (+ headway/leader)": scores(te[TARGET][bad], pb[bad.values]),
                  "GBDT (+ weather)": scores(te[TARGET][bad], pw[bad.values])}))

    print("\n=== is the knock-on correlation confounded by weather? ===")
    sub = df[df["headway_s"] < 300].dropna(
        subset=["leader_entry_delay", "follower_delay_gained"] + VARS)
    x, y = sub["leader_entry_delay"].values, sub["follower_delay_gained"].values
    print(f"n={len(sub):,}  raw r = {np.corrcoef(x, y)[0, 1]:+.4f}")
    Z = pd.get_dummies(sub[VARS], columns=["weather_code"], drop_first=True).astype(float).values
    hours = pd.get_dummies(sub["hour"], drop_first=True).astype(float).values
    for lab, M in (("weather", Z), ("weather + hour", np.column_stack([Z, hours]))):
        M = np.column_stack([M, np.ones(len(M))])
        rx = x - M @ np.linalg.lstsq(M, x, rcond=None)[0]
        ry = y - M @ np.linalg.lstsq(M, y, rcond=None)[0]
        print(f"  partial r | {lab:<16} = {np.corrcoef(rx, ry)[0, 1]:+.4f}")

    print("\n=== daily aggregates: any weather signal at all? ===")
    day = df.groupby("date").agg(entry=("entry_delay", "mean"),
                                 gain=("follower_delay_gained", "mean"),
                                 snow=("snowfall", "sum")).reset_index()
    r_all = np.corrcoef(day["snow"], day["entry"])[0, 1]
    z, se = np.arctanh(r_all), 1 / np.sqrt(len(day) - 3)
    d2 = day[day["date"] != "09JAN2025"]
    print(f"  r(daily snowfall, mean entry_delay) = {r_all:+.3f}  "
          f"95% CI [{np.tanh(z - 1.96 * se):+.3f}, {np.tanh(z + 1.96 * se):+.3f}]")
    print(f"  dropping 09JAN2025 (the one big snow day) = "
          f"{np.corrcoef(d2['snow'], d2['entry'])[0, 1]:+.3f}")
    sd = day[day["snow"] > 0]
    print(f"  mean entry_delay: {len(sd)} snow days {sd['entry'].mean():.0f}s "
          f"vs {len(day) - len(sd)} dry days {day[day['snow'] == 0]['entry'].mean():.0f}s")


def main() -> None:
    if CACHE.exists():
        w = pd.read_csv(CACHE, dtype={"ptcar": str})
        print(f"weather from cache: {len(w):,} rows  ({CACHE.name})")
    else:
        print("fetching Open-Meteo hourly weather for the 5 junction points:")
        w = fetch()
        w.to_csv(CACHE, index=False)
        print(f"-> {CACHE}")
    profile(w)
    evaluate(w)


if __name__ == "__main__":
    main()
