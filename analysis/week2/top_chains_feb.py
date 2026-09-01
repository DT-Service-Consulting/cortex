"""Top-10 longest delay chains in February 2025, using the cascade_check.py definition."""
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "junction" / "traversals_202502.csv"
HW_MAX, THR = 300, 60

def secs(t):
    if not isinstance(t, str) or not t.strip():
        return float("nan")
    h, m, s = (int(x) for x in t.split(":"))
    return h*3600 + m*60 + s

df = pd.read_csv(P, dtype=str)
df["entry_s"] = df["p1_real_dep"].map(secs)
for c in ("entry_delay", "exit_delay", "delay_gained"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["entry_s", "entry_delay", "exit_delay", "delay_gained"])
df = df.sort_values(["date", "direction", "tunnel_track", "entry_s"])
df["moved"] = df["delay_gained"] > THR
df["service"] = df["relation"].str.strip().str.split().str[0]

chains = []
for (date, direction, track), g in df.groupby(["date","direction","tunnel_track"], sort=False):
    g = g.reset_index(drop=True)
    hw = g["entry_s"].diff()
    linked = (hw <= HW_MAX) & g["moved"] & g["moved"].shift(1).fillna(False)
    cid = (~linked).cumsum()
    for _, run in g.groupby(cid):
        run = run[run["moved"]]
        if len(run) >= 2:
            chains.append((len(run), run["delay_gained"].sum(), date, direction, track, run))

chains.sort(key=lambda x: (x[0], x[1]), reverse=True)
print(f"February 2025: {len(df):,} traversals, {int(df['moved'].sum()):,} moved (>{THR}s), "
      f"{len(chains):,} chains of 2+\n")

def hhmm(s):
    s = int(s) % 86400
    return f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"

for i, (L, tot, date, direction, track, run) in enumerate(chains[:10], 1):
    print(f"--- #{i}  len {L}  |  {date}  {direction}  track {track}  |  total gained {int(tot):,}s")
    prev = None
    for _, r in run.iterrows():
        hw = "" if prev is None else f"{int(r.entry_s-prev):>4}s"
        print(f"      {hhmm(r.entry_s)}  train {r.train_no:>5}  {r.service:<5} "
              f"hw {hw:>5}  entry {int(r.entry_delay):>5}s  exit {int(r.exit_delay):>5}s  "
              f"gained {int(r.delay_gained):>5}s")
        prev = r.entry_s
    print()
