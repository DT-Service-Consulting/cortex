# week2 — the February top-10 delay chains

Focused working set: the 10 longest delay cascades in February 2025, with the
timetable data needed to analyse each one.

Regenerate everything with `python analysis/week2/chains.py`.

## What a chain is

Restated from `analysis/causal/cascade_check.py`, with the parameters explicit
because they are free choices, not facts:

| ingredient | rule |
|---|---|
| **scope** | trains sharing `(date, direction, tunnel_track)` — same physical tunnel track, same direction, same day — sorted by entry time |
| **moved** | `delay_gained > THR`, `THR = 60 s`, where `delay_gained = p5_delay_arr − p1_delay_dep` |
| **link** | train *i* links to *i−1* iff **both** are moved **and** entry headway ≤ `HW_MAX = 300 s` |
| **chain** | walk the sorted sequence; every broken link starts a new chain; report chains of ≥ 2 |
| **selection** | top 10 by length, ties by total `delay_gained` |

`tunnel_track` is `LINE_NO_DEP` read at Brussels-Central. It is not a raw
"platform" column — the Infrabel feed has none. Inside the junction `LINE_NO_DEP`
holds `0/1`–`0/6`, the six tunnel tracks, values that occur nowhere else in
Belgium; odd = S2N, even = N2S. See `data/README.md` and `data/fetch_months.py:112`.

**Known brittleness.** The link uses a hard threshold on both ends, so a cascade
running through a train that lands just under 60 s is split into two chains.
Lengths are a **lower bound**. The per-chain extracts therefore carry the
sub-threshold trains that bounded each chain (`role = breaker_before` /
`breaker_after`) — those rows are what answer *why did it start here* and *why
did it stop here*.

## Where delay is lost

`delay_gained` decomposes **exactly** (verified: max residual 0 s over all 23,601
February traversals) into seven additive terms:

```
run p1->p2 | dwell p2 | run p2->p3 | dwell p3 | run p3->p4 | dwell p4 | run p4->p5
```

Central (215) is always `p3` in both directions (`build_traversals.ORDER`), so
`dwell_p3_central` is time lost at Gare Centrale and `run_p2_p3` is time lost on
the approach to it. `p2` and `p4` are Congrès and Chapelle, "stop in open track"
points where arrival and departure times are usually identical — their dwell
terms are ≈0 by construction.

A dwell term is `delay_dep − delay_arr` = **actual − planned** dwell, i.e. it is
*excess* dwell and can legitimately be negative when a late train recovers its
scheduled stand. `p3_planned_dwell_s` and `p3_actual_dwell_s` are carried so this
is readable. (Example: chain `20250219-N2S-0_2-1714` has six consecutive trains
at exactly −60 s — all have `real_arr == real_dep` against a 60 s planned stand.
Real behaviour, not a clock artifact.)

Over the 103 trains in the top 10: mean `run_p2_p3` **+229.0 s**,
`dwell_p3_central` **+47.7 s**, `run_p3_p4` **−19.6 s**. Central dwell is present
in 86% of trains but is only 15% of total gain — the queue on the approach is the
bulk of it.

## Files

Committable (small, and the definition of the working set):

| file | rows | what |
|---|---|---|
| `chains.py` | — | the definition and the extractor |
| `chains_top10.csv` | 10 | manifest, one row per chain |
| `top_chains_feb.py` | — | human-readable printout of the same 10 chains |
| `where_lost.py` | — | 7-leg attribution, all traversals / chains / starters / followers |

Bulk extracts under `data/week2/` (`data/` is local-only, nothing there is
committed):

| file | rows | what |
|---|---|---|
| `chain_members.csv` | 103 | one row per (chain, position, train), with all 7 legs |
| `timetable/<chain_id>.csv` | 70–120 | long-form 5-point timetable, one row per train per junction point |
| `control_days.csv` | 268 | the same cast on every other February day |

### `chains_top10.csv`

`chain_id`, `rank`, `date`, `iso_date`, `direction`, `tunnel_track`, `length`,
`start_time`, `end_time`, `span_s`, `total_gained_s`, `max_gained_s`,
`mean_headway_s`, `min_headway_s`, `starter_train`, `terminator_train`,
`members`, `service_mix`, `recurrence_key`, `cast_cluster`, `thr_s`, `hw_max_s`,
`group_key`, `selection`.

**Join on `chain_id`, not `rank`.** `chain_id` is content-derived
(`20250219-N2S-0_2-1714` = date, direction, track, start time) and stable; `rank`
is positional and silently means something else if `THR` or `HW_MAX` changes. The
parameters are stamped into the file for the same reason.

### `timetable/<chain_id>.csv`

Drawn from the **long form** (`junction_traversing_202502.csv`), a strict superset
of the wide form: it keeps per-point `LINE_NO_DEP`/`LINE_NO_ARR` (so the ~1.45% of
trips whose endpoint track differs from Central's are visible), per-point
`THOP1_COD`, and the `*_DATE_*` columns. Five rows per train, in travel order.

Added columns: `chain_id`, `role`, `chain_position`, `entry_delay`, `exit_delay`,
`delay_gained`, `entry_s`.

`role` ∈ `member` · `breaker_before` · `breaker_after` · `context` (other
traversals on the same track and direction within ±30 min).

### `control_days.csv`

The counterfactual, and the reason this is more than 103 trains. The same
scheduled cast runs the same track every February weekday; on most days there is
no long chain. One row per (chain, other date).

**Filter on `is_weekday` before averaging.** All 10 chain days are weekdays
(Mon–Fri), and at weekends only 5–7 of a 12-train cast runs at all, so weekend
rows drag the raw counts down. Use `cast_moved_rate` (`cast_moved / cast_ran`)
rather than `cast_moved` — the denominator swings 5→12 across days.

Two chain-length columns, and they differ on 148 of 268 rows:
`max_chain_len_in_window` is restricted to chains starting within ±30 min of the
reference chain, `max_chain_len_that_day` is the whole day on that track. The
windowed one is what answers "how often does *this* cascade recur"; the day-wide
one will happily report a 08:00 cascade for a 17:14 chain.

Example — `20250207-N2S-0_2-1714`, over the 19 other February weekdays: the cast
averages 9.1 trains running, **5.7 moved (rate 0.55)**, 1,250 s of gain, and a
windowed max chain length of 4.8. On the chain day: 12 of 12 moved (rate 1.00),
3,817 s, chain length 12.

Worth noting before treating "chain day" as anomalous: the morning S2N `0/3`
cluster already runs at a control moved-rate of **0.74–0.77** and a windowed
chain length above 5 on an ordinary weekday. Those services are chronically
cascading; the top-10 day is the tail of a routine condition, not an isolated
incident. The evening N2S `0/2` cluster sits lower at 0.54–0.56.

## Recurrence

These are not 10 independent incidents. Exact cast equality finds nothing (the
cast varies day to day), but Jaccard overlap on the member sets is 0.64–0.83 and
resolves into four clusters (`cast_cluster`):

- **C1** — 4 chains, N2S track `0/2`, ~17:14–17:27, on 07/11/12/19 Feb. Always
  present: 14, 438, 2087, 2839, 3239, 3738, 8007.
- **C2** — 4 chains, S2N track `0/3`, ~08:36–08:42, on 05/13/14/20 Feb. Always
  present: 1507, 1707, 1907, 2409, 6557, 9223.
- **C3** — 1 chain, N2S `0/6`, 03 Feb 08:47.
- **C4** — 1 chain, N2S `0/2`, 14 Feb 07:51.

Eight of the ten are two recurring peak-hour patterns. That is the more useful
framing than "ten worst days": the same scheduled services on the same two tracks
fail repeatedly, which is what makes the control-day comparison possible.
