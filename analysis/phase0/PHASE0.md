# Phase 0 — observational facts and a floor

Run: `cd analysis/phase0 && python step1_pairs.py && python step2_baselines.py && python step3_dekker.py`
Data: January 2025, 29,997 Nord↔Midi traversals. Outputs in `data/phase0/`.

## Step 1 — leader/follower pairs

29,685 pairs of consecutive traversals sharing `(date, direction, tunnel_track)`.
Median headway **5.3 min**.

**A leakage trap, resolved.** Median headway (5.3 min) is *shorter* than median
traversal time (8.8 min), so the leader is usually **still inside the junction**
when the follower enters. `leader_exit_delay` is therefore not knowable at
prediction time. It is kept in the table as `leader_exit_delay_POSTHOC` for
descriptive use only; models use `leader_entry_delay`.

This costs a lot of apparent signal, and that is the honest picture:

| Headway | r(leader **exit**, follower gain) | r(leader **entry**, follower gain) |
|---|---|---|
| < 5 min | 0.275 | **0.119** |
| 5–10 min | 0.106 | 0.015 |
| 10–20 min | 0.046 | 0.023 |
| 20–60 min | 0.088 | 0.073 |

The 0.275 quoted earlier is post-hoc. The usable figure is 0.119.

## Steps 2–4 — task, split, baselines

Task: at junction entry, predict `exit_delay` (seconds). Temporal split, train
days 1–24 (22,817 rows), test days 25–31 (6,868 rows).

| model | MAE | RMSE | precision | recall |
|---|---|---|---|---|
| zero | 171.4 | 380.3 | — | 0.000 |
| **persistence** | **60.2** | **99.4** | 0.960 | 0.653 |
| mean-gain | 63.1 | 94.1 | 0.919 | 0.694 |
| GBDT (no interaction) | 66.1 | 143.2 | 0.931 | 0.698 |
| **GBDT (+ headway/leader)** | **57.4** | 127.1 | 0.937 | 0.710 |

Three results worth carrying forward:

1. **Persistence is hard to beat**, exactly as Nguyen et al. found. A GBDT with
   entry delay, hour, track, direction and service but *no interaction features*
   **loses to it** by 9.7% MAE. Feature richness alone is not enough.
2. **Only the headway/leader block beats the baseline.** It is worth +8.63 s MAE
   (13.1%) and is the entire source of the win. The one feature block with a
   causal story is the one carrying signal — which is the argument for the whole
   causal programme, arrived at empirically.
3. **The win is MAE-only; RMSE gets worse** (127.1 vs 99.4). See the diagnosis
   below — the cause is *not* what it first looked like.

Short-headway subgroup (<5 min, n=3,208), where interaction should matter most:

| model | MAE | RMSE |
|---|---|---|
| persistence | 86.2 | 129.9 |
| GBDT (no interaction) | 80.3 | 123.4 |
| GBDT (+ headway/leader) | **72.1** | **110.0** |

Here the interaction model wins on **both** MAE and RMSE — 16% better MAE than
persistence. The knock-on mechanism is real and concentrated where theory says it
should be.

## Diagnosing the RMSE loss

An earlier draft of this note claimed the model "cannot predict large delays".
**That was wrong.** Three checks:

- **Distribution shift: ruled out.** Train days 1–24 vs test days 25–31 are
  near-identical (mean 163.4 vs 159.5 s, p90 406 both, fraction >300 s 0.147 vs
  0.148).
- **Persistence's precision 0.960 is genuine**, not a threshold artifact:
  TP=665, FP=28, FN=354, TN=5,821.
- **Test RMSE by headway band localises the damage entirely to one regime:**

| Headway | n | persistence RMSE | GBDT+ RMSE |
|---|---|---|---|
| < 5 min | 3,208 | 129.9 | **110.0** |
| 5–10 min | 2,390 | 60.3 | 72.2 |
| 10–20 min | 1,059 | 64.9 | 79.9 |
| **> 20 min** | 211 | 52.5 | **501.2** |

The model is *better* on short headways and blows up on long ones. Long headways
are only **3.7% of training rows**, so it is extrapolating into a regime it has
barely seen — and those 211 rows dominate the global RMSE.

**The fix comes straight from Li et al. 2024:** above ~20 min headway train
interaction does not exist, so interaction features there are noise. Gating —
use the interaction model below 20 min, persistence above — gives:

| fold | persistence MAE / RMSE | GBDT+ | hybrid |
|---|---|---|---|
| train ≤17, test 18–24 | 55.5 / 89.6 | 52.2 / 105.5 | 52.1 / 105.5 |
| train ≤24, test 25–31 | 60.2 / 99.4 | 57.4 / 127.1 | **56.0 / 92.4** |

In the second fold the hybrid beats persistence on **both** MAE (−7.0%) and RMSE.
In the first it does not — gating changes nothing there, so that fold's RMSE gap
has a different, still-unidentified cause. **RMSE remains an open issue**;
treat the MAE gain as established and the RMSE story as partially explained.

**Rolling-origin check:** the MAE margin over persistence is stable across folds
(+6.1%, +4.6%; hybrid +6.1%, +7.0%), so it is not an artifact of one test week.

## Step 5 — Dekker's exponential decay: mostly falsified

Claim: a station sheds delay proportionally, `dD/dt = −B_i·D_i`, so the change in
a train's delay should be linear in delay carried, with negative slope.

Dwell rows are restricted to trains that actually stop (`stop_type = '='`);
Congrès and Chapelle are 95–98% pass-through, where `arr == dep` by construction
and the regression would be definitionally zero.

**4 of 17 segments are consistent; 0 contradict; 13 are flat.**

- **Consistent** — dwell at the two terminals only: Midi (−0.058 S2N, −0.034 N2S)
  and Nord (−0.070 S2N, −0.052 N2S), r ≈ −0.2.
- **Flat** — dwell at Central (+0.008, +0.004) and **every run segment**
  (|slope| ≤ 0.019).

Interpretation: between points, delay changes by a roughly **constant offset**
(see the intercepts: +6.2, +8.8, −17.1, −25.3 s) that is *independent of the delay
carried*. That is **additive, not multiplicative** — the diffusion model's core
mechanism does not operate at junction scale, except at the two big terminals
where long dwells give trains room to recover proportionally.

This is a useful negative result: it says a proportional-decay term is the wrong
functional form here, and a constant per-segment offset plus a terminal recovery
term is the right skeleton. **It is also a concrete calibration target for
SUMO** — the simulator should reproduce flat run segments and shedding terminals,
not uniform proportional decay.

## Phase 1 gate, restated with numbers

SUMO must reproduce: per-station median delay curve including the exit drop;
`delay_gained` median ≈ +3 s / mean ≈ +27 s with p99 ≈ +350 s; headway median
5.3 min; **flat run slopes and terminal dwell slopes ≈ −0.03 to −0.07**; and a model
trained on simulated traversals should land **near** MAE ≈ 56–60 / RMSE ≈ 92–99
when tested on real data. If simulated data makes the task markedly *easier*,
the simulator is missing the noise that makes reality hard — that is a failure,
not a success.

## Step 7 — weather (negative result)

`step5_weather.py`. Replicates RIDE's own recipe: Open-Meteo Historical Weather
API, per operational point, `Europe/Brussels`, the same six hourly variables
(`temperature_2m`, `rain`, `snowfall`, `relative_humidity_2m`, `wind_speed_10m`,
`weather_code`), attached at the entry point and entry hour — the prediction
moment. 100% join rate.

January 2025 had real variation: mean 2.8 °C, **200 sub-zero hours**, 174 rain
hours, 38 snowfall hours. So the month is not too quiet to test.

**Weather does not help.**

| model | MAE | RMSE |
|---|---|---|
| persistence | 60.2 | 99.4 |
| GBDT (+ headway/leader) | **57.4** | 127.1 |
| GBDT (+ weather) | 58.4 | 129.7 |

The weather block is worth **−0.99 s MAE (−1.7%)** — six added features, slightly
*worse*. It does not help in the bad-weather subgroup either (rain > 0.5 mm, snow,
or sub-zero; n = 855): 49.2 vs 48.0 MAE.

**Weather does not confound the knock-on signal either.** Partial correlation of
`leader_entry_delay` against `follower_delay_gained` at headway < 5 min:

```
raw                        r = +0.1195
| weather                  r = +0.1217
| weather + hour           r = +0.1151
```

Unchanged. Weather was the most plausible latent common cause behind the
leader/follower correlation; **it is not it.** That strengthens the case that the
coupling is a genuine track-occupancy mechanism, not shared exposure.

**The one apparent signal is a single day.** Daily snowfall vs mean entry delay
gives r = +0.414 (95% CI [+0.07, +0.67], n = 31) — but dropping 09JAN2025, the one
heavy snow day, collapses it to **r = −0.007**. Snow days average 138 s entry delay
against 132 s on dry days: a 6-second difference.

**Why this is the expected answer.** The junction is a 3.5 km sheltered tunnel and
a train is inside it for ~9 minutes. Weather acts on the *network*, and whatever it
does arrives already encoded in `entry_delay`, which is the first feature the model
has. There is little room for weather to act on the traversal itself.

**Do not over-read the negative.** One month, one junction, six snow days of which
one matters. This says weather adds nothing *to this model at this scale* — not that
weather is irrelevant to Belgian rail delay. Testing it properly needs RIDE's
2023–2025 span. Also note Open-Meteo's grid is coarser than the junction: all five
points return **identical** values (max spread 0.00 °C), so per-point querying buys
nothing here.

---

## Step 9 — full year 2025: what survives, what does not

`step7_full_year.py`. All 12 months fetched via `data/fetch_months.py`
(**338,914 traversals**, 335,323 pairs). Weather extended to the full year.

**A download bug worth recording.** Three months (Aug/Sep/Oct) initially arrived
truncated — a 342 MB file returned as 1 MB, ending the read loop cleanly with no
error, yielding one day of data instead of a month. `download()` now verifies
received bytes against `Content-Length` and retries. Any future fetch must keep
that check; a silently partial month is invisible downstream.

### January was the best month of the year

| | `delay_gained` p50 | mean | p99 |
|---|---|---|---|
| **January** | **3 s** | **27.9 s** | 348 s |
| worst (October) | 12 s | 48.3 s | 503 s |
| range across months | 3–13 s | 27.4–48.3 s | 348–503 s |

Phase 0's headline numbers came from an unusually good month. Nothing is wrong
with them; they simply are not representative.

### Survives: the knock-on correlation

Per-month r(leader entry delay, follower delay gained) at headway < 5 min:
**mean +0.110, sd 0.018, range [+0.086, +0.148]** across 12 months. January's
+0.119 was typical. Stable, and the full-year partial correlations barely move:

```
raw                        +0.1099
| weather                  +0.1086
| weather + hour           +0.1071
| weather + hour + month   +0.1050
```

Controlling for weather, time of day *and* month leaves it essentially intact.
**The confounders we can measure do not explain it.**

### Downgraded: the model's advantage

Expanding-window rolling origin, 9 folds:

- MAE: **mean +4.0% over persistence, sd 4.9, beats it in 8 of 9 months** — but
  loses outright in July (−5.9%). January's +6.1%/+4.6% was optimistic.
- RMSE: **worse in all 9 folds**, often badly (July 94.2 → 189.4).

In the January write-up the RMSE gap was called "partly unexplained" and blamed on
the sparse >20 min headway band. With nine folds that reading no longer holds: the
degradation is **systematic, not a fold artifact**. The likely driver is
month-to-month non-stationarity — `gain_mean` ranges 27–48 s across the year, so a
model trained on earlier months carries the wrong level into later ones. That is
an argument for time-varying estimation (Haslbeck's mgm) rather than one static fit.

### Confirmed dead: weather

Six folds, weather block added to the full feature set:
**mean −0.07 s MAE (sd 0.68), helps in 2 of 6 months.** Noise around zero.

Full-year daily snowfall vs mean entry delay: **r = +0.088, 95% CI [−0.015,
+0.189]** over 365 days with 15 snow days — CI includes zero. Snow days average
150 s entry delay against 142 s on dry days. The January r = +0.414 was, as
suspected, one day of leverage.

**Weather is settled: it adds nothing at junction scale, and it is not the
confounder.** This was the largest open caveat in Phase 0 and it is now closed
with a year of data rather than a month.
