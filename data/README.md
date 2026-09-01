# data/

Local-only (excluded via `.git/info/exclude`). Nothing here is committed.

## Contents

| Path | What |
|---|---|
| `2025_raw.csv` | Link manifest: month → download URL, 12 rows. `;`-separated, UTF-8 BOM. |
| `raw/Data_raw_punctuality_202501.csv` | January 2025 raw export, 329.5 MB, 1,965,004 rows. |
| `junction/junction_202501.csv` | North-South junction subset, 27.3 MB, 152,601 rows (7.77%). |
| `junction/junction_traversing_202501.csv` | Traversing trains only, long form, 28.4 MB, 149,985 rows. |
| `junction/traversals_202501.csv` | One row per Nord↔Midi traversal, wide form, 10.5 MB, 29,997 rows. |
| `reference/operational_points.csv` | All 1,359 Infrabel operational points with PTCAR id, names, class, lat/lon. |
| `extract_junction.py` | Streams a raw monthly file → junction subset + profile. ~7 s per month. |
| `build_traversals.py` | Junction subset → per-traversal table + propagation profile. |

Source: Infrabel open data, dataset `stiptheid-gegevens-maandelijksebestanden`
(151 monthly files available). Reference points from
`operationele-punten-van-het-netwerk`.

## Raw schema (21 columns, comma-separated)

Careful: the *manifest* is `;`-separated, the *data* is `,`-separated.

- **Identity** — `DATDEP` (`01JAN2025`), `TRAIN_NO`, `RELATION` (`IC 31`),
  `RELATION_DIRECTION`, `TRAIN_SERV`, `CIRC_TYP`
- **Location** — `PTCAR_NO` + `PTCAR_LG_NM_NL` (**Dutch** name only),
  `LINE_NO_DEP`, `LINE_NO_ARR`, `THOP1_COD`
- **Timing** — `PLANNED_TIME_ARR/DEP`, `REAL_TIME_ARR/DEP`, matching `*_DATE_*`
  columns, and precomputed `DELAY_ARR` / `DELAY_DEP`

`DELAY_*` is **in seconds and signed** (negative = early). January medians sit at
~54 s; 13.2% of arrivals are more than 5 minutes late — that percentage is over
the 146,481 non-null rows, as 6,120 of the 152,601 junction rows have a blank
`DELAY_ARR` (6,003 blank for `DELAY_DEP`). Those blanks are patterned, not
scattered — see "Where the blanks are" below.

`THOP1_COD` marks the stop type: `=` a commercial stop (arrival ≠ departure),
`D` a pass-through (arrival == departure), `P` origin/terminus, plus a few rare
`( ) >` codes.

## The junction

`PTCAR_NO` uses the same numbering as `STATIONS` in
`core/simulation/build_hard.py`, confirmed against the operational-points
reference:

| PTCAR | Symbolic | Name (FR) | Data name (NL) | Class | lat, lon |
|---|---|---|---|---|---|
| 215 | FBCL | BRUXELLES-CENTRAL | BRUSSEL-CENTRAAL | Station | 50.84531, 4.35676 |
| 216 | FBCO | BRUXELLES-CONGRES | BRUSSEL-CONGRES | Stop in open track | 50.85178, 4.36222 |
| 217 | FBCK | BRUXELLES-CHAPELLE | BRUSSEL-KAPELLEKERK | Stop in open track | 50.84118, 4.34784 |
| 220 | FBMZ | BRUXELLES-MIDI | BRUSSEL-ZUID | Station | 50.83580, 4.33605 |
| 221 | FBN | BRUXELLES-NORD | BRUSSEL-NOORD | Station | 50.85960, 4.36125 |

**Trip structure.** 29,997 of the 32,613 January trips produce exactly 5 rows —
one per junction point. The remaining 2,616 are single-row, and they sit almost
entirely at **Midi** (220: 2,532; 221: 84; none at 215/216/217) — trains that
touch the junction's southern end without traversing it. Note that `THOP1_COD=P`
is a separate thing: 4,505 rows carry it, including rows *inside* full 5-row
trips, so a train can be origin/terminus at Central and still have all five
junction records.

Row order in the file **is** traversal order, and only two orders occur, evenly
split: `220→217→215→216→221` (S2N, 14,964 trips) and `221→216→215→217→220`
(N2S, 15,033). So the subset is already a clean per-train ordered sequence
through the junction — the natural input for delay-propagation and
causal-discovery work, each train giving five successive (planned, actual,
delay) observations with direction recoverable from the row order alone.

Example, train 3117 southbound→northbound on 01JAN2025:

```
 220 BRUSSEL-ZUID         plan 17:55:00  real 17:53:22  arr -97  dep +19  (=)
 217 BRUSSEL-KAPELLEKERK  plan 18:01:00  real 18:01:31  arr +31  dep +31  (D)
 215 BRUSSEL-CENTRAAL     plan 18:03:00  real 18:02:51  arr  -8  dep +44  (=)
 216 BRUSSEL-CONGRES      plan 18:06:00  real 18:06:14  arr +14  dep +14  (D)
 221 BRUSSEL-NOORD        plan 18:08:00  real 18:08:21  arr +21  dep +39  (=)
```

**Track-level detail.** Inside the junction, `LINE_NO_DEP` takes values
`0/1`–`0/6` — the six tunnel tracks. Every one of the 29,997 five-row trips uses
a **single** tunnel track across all of 215/216/217 (checked directly: distinct
track count per trip is 1 in every case), splitting `0/2` 5,890 · `0/4` 5,767 ·
`0/1` 5,673 · `0/3` 5,502 · `0/5` 3,789 · `0/6` 3,376. At Midi (220) and Nord
(221) the value switches to real line numbers
(`50`, `50A`, `50C`, `96`, `96N`, `25`, `27`, `36N`, `124`, `161/2`) where trains
enter or leave the tunnel. That gives per-track occupancy, which is what the
SUMO model in `core/simulation/` needs to be calibrated against.

## Traversal layer (the working set)

`build_traversals.py` reduces the junction subset to **trains that pass through
both Nord (221) and Midi (220)** — i.e. that actually traverse the junction —
one row per traversal:

- **29,997 traversals** kept, 2,616 trips dropped (`does_not_traverse`). Every
  dropped trip is a single-station touch, 2,532 of them at Midi.
- Zero fell into `not_five_points` or `unexpected_order`: the structure is
  completely regular, so no special-casing is needed downstream.
- Near-perfect direction balance: **N2S 15,033 / S2N 14,964**.
- Tunnel track split: `0/2` 5,890 · `0/4` 5,767 · `0/1` 5,673 · `0/3` 5,502 ·
  `0/5` 3,789 · `0/6` 3,376.

The filtered set exists in **two shapes**, same 29,997 trips in both (trip keys
verified identical):

- **Long** — `junction/junction_traversing_202501.csv`, 149,985 rows (29,997 × 5).
  The original 21 raw columns untouched, plus `direction`, `seq` (1–5, travel
  order) and `tunnel_track`. Rows are grouped by trip and ordered along travel;
  trips are ordered by entry time. Use this for row-level / sequence work.
- **Wide** — `junction/traversals_202501.csv`, 29,997 rows, one per traversal.
  Use this for one-sample-per-traversal modelling.

Output `junction/traversals_202501.csv` (10.5 MB). Columns: `date`, `train_no`,
`relation`, `relation_direction`, `direction`, `tunnel_track`, `entry_line`,
`exit_line`, then `p1_*`…`p5_*` (ptcar, name, planned/real arr+dep, delay
arr+dep, stop type) **ordered along the direction of travel**, plus
`entry_delay`, `exit_delay`, `delay_gained`.

`exit_line` is blank for 4,794 traversals (4,361 of them N2S, i.e. exiting at
Midi) — it goes missing together with the exit departure fields, see below.

Positions are travel-ordered, not station-fixed: `p1` is Midi for S2N and Nord
for N2S. That makes `p1_delay_arr … p5_delay_arr` a directly usable variable
sequence for propagation models — samples = traversals, variables = position
along the junction.

### What the delays look like

Median arrival delay by position (seconds):

```
S2N:  midi 42  ->  chapelle 55  ->  central 64  ->  congres 81  ->  nord 74
N2S:  nord 26  ->  congres 30  ->  central 51  ->  chapelle 60  ->  midi 47
```

Two things stand out. Delay **accumulates monotonically through the tunnel** and
then **drops at the exit station** (congres 81 → nord 74; chapelle 60 → midi 47),
which is the signature of recovery margin padded into the final approach — worth
knowing before treating the last hop as a normal propagation step.

Delay gained across the whole junction (`exit_delay - entry_delay`, seconds):

```
S2N: n=14,964 p10=-47 p50=+4 mean=+30 p90=+139 p99=+367
N2S: n=15,033 p10=-44 p50=+2 mean=+26 p90=+127 p99=+326
```

Median ≈ 0 but mean ≈ +28: the junction is neutral for a typical train and the
average is driven by a heavy right tail. Any model fit on the mean will be
fitting the tail, so **model the distribution, not the average**. Entry delay is
also asymmetric — trains enter from Midi noticeably later (median 42 s) than from
Nord (26 s).

### Where the blanks are

Blank delays occur at exactly two places: `p1_delay_arr` (4,741 rows) and
`p5_delay_dep` (4,789 rows) — the *arrival* side of the entry point and the
*departure* side of the exit point. No middle position has a single blank, and
`p1_delay_dep` / `p5_delay_arr` are never blank.

These blanks come as a bundle: on a p5-blank row, `THOP1_COD`, `PLANNED_TIME_DEP`,
`REAL_TIME_DEP` and `LINE_NO_DEP` are all empty together (4,701 of 4,720), while
the arrival fields are fully populated. So it is one "no onward record at this
point" class, concentrated at Midi (4,336 of 4,720).

**Do not read it as termination.** Some of those trains genuinely end at
Brussel-Zuid (`IC 35: ROTTERDAM CENTRAAL -> BRUSSEL-ZUID`), but others plainly
continue (`IC 06-1: BRUSSELS AIRPORT - ZAVENTEM -> TOURNAI`, `IC 29: DE PANNE ->
LEUVEN`). Nor is it the `THOP1_COD = P` origin/terminus code — no Midi row ever
carries `P`. Treat it as a recording gap of unknown cause until someone confirms
it with Infrabel.

Because `delay_gained` is built from `p1` **departure** and `p5` **arrival** — the
two sides that are never blank — it is present for **all 29,997 traversals**. No
imputation or row-dropping is needed for the headline propagation variable.

One real gotcha: **59 traversals cross midnight**, and `DATDEP` stays on the
departure date. Note also that **hours are not zero-padded** in this feed
(`0:01:11`, `9:57:00`), so string comparison of times is wrong — `"9:57:00" >
"10:03:00"` is True. Parse to seconds; `build_traversals.py` has a `secs()`
helper. The precomputed `DELAY_*` columns already handle both issues.

## Train-side mapping

No join needed: `TRAIN_NO` carries `RELATION` (`IC 31`) and `RELATION_DIRECTION`
(`IC 31: CHARLEROI-CENTRAL -> ANTWERPEN-CENTRAAL`) on every row, so service and
origin/destination come for free. January's junction traffic spans 1,385
distinct train numbers, and the top relations mix service types — `P` (peak-hour
extras, 6,228 rows), `L B1-2` (6,013), `EURST` (Eurostar, 5,859), `IC 06-2`,
`IC 01`, `IC 23-1`, `IC 31` — so an IC / L / P / international split is available
straight from `RELATION`.

## Naming mismatch

The open data calls the column `PTCAR_NO`; the Azure SQL gold layer
(`sqldb-gold-cortex.session.sql`, `core/punctuality.py`) calls it `PTCAR_ID`.
Same identifier space — reconcile before joining against
`infra_operational_points`.
