# Causal analysis — knock-on delay at the junction

Run: `python analysis/causal/run_causal.py`. Data: 313,151 leader/follower pairs,
2025 full year. All continuous variables rank-transformed to normal scores (the
delay tail breaks Gaussian partial correlations).

**Question.** When a train enters close behind another, does the leader's state
*cause* the follower to lose time, or do they merely share upstream causes?

**Two statistical hazards, handled explicitly.** At n = 313k every independence
test rejects, so significance is uninformative — discovery uses subsampling with
stability selection and everything is reported as effect size. And heavy tails
make raw partial correlations unreliable — hence the rank transform.

## Design A — constraint-based discovery with background knowledge

10 variables in 4 temporal tiers (fixed-before-the-day → leader enters → follower
enters → outcome), 29 arrows forbidden by tier. PC and FCI, 40 subsamples of
n = 4,000 each. FCI matters because it admits latent confounders rather than
inventing direct edges for them.

| edge | PC | FCI |
|---|---|---|
| `act_hw → foll_gain` | **100%** | **100%** |
| `leader_entry → act_hw` | 100% | 100% |
| `leader_entry → foll_entry` | 100% | 100% |
| `sched_hw → act_hw` | 100% | 100% |
| **`leader_entry — foll_gain`** (direct) | **45%** | **22%** |

**Headway is a stable direct cause of delay gained. The direct leader→follower
edge is not stable** — and it survives even less under FCI, i.e. once latent
confounding is allowed for. The recovered structure routes the leader's influence
through headway: `leader_entry → act_hw → foll_gain`.

## Design B — negative control (the sharpest test available)

The two explanations make *opposite* predictions. A physical track-occupancy
mechanism must weaken as trains separate. Shared upstream disruption should
persist at all headways, because a disruption hits near and far trains alike.

| headway | n | raw r | partial r (given hour, month, weather) |
|---|---|---|---|
| < 3 min | 49,073 | **0.243** | 0.243 |
| 3–5 min | 87,944 | 0.189 | 0.189 |
| 5–10 min | 112,003 | 0.074 | 0.074 |
| 10–20 min | 50,954 | 0.026 | 0.026 |
| 20–40 min | 12,195 | 0.008 | 0.012 |
| 40–120 min | 982 | 0.005 | −0.001 |

**Monotone decay to zero — the mechanism prediction, not the confounding one.**
And partial ≈ raw throughout: the confounders we can measure explain none of it.

## Design C — instrumental variable

Scheduled headway comes from a timetable fixed months ahead, so it cannot be caused
by today's disruption, yet it strongly drives actual headway. Controls: hour, month,
track, direction. First stage: `d(actual)/d(scheduled) = +0.434`, partial
R² = 0.356, **F = 172,709**.

| estimator | effect of +1 min headway on delay gained |
|---|---|
| OLS | −4.23 s |
| **2SLS** | **−1.81 s**  (bootstrap 95% CI **[−2.06, −1.55]**) |

**One extra minute of headway causally reduces delay gained by ≈1.8 s.** OLS
overstates the magnitude by ~2.3×, in the expected direction: disruption both
compresses headway and raises delay, so the naive slope absorbs the confounding.

## Mediation — how much of the leader effect goes through headway?

Headway < 5 min, n = 137,017:

```
r(leader_entry, foll_gain)                    +0.2922
  | context (hour, month, weather)            +0.2925
  | context + act_hw                          +0.1995
  | context + act_hw + foll_entry             +0.1496
```

Conditioning on headway removes about a third of the association; the follower's
own entry delay removes more. **A residual +0.15 remains: this is partial, not
complete, mediation.** That is consistent with the unstable-but-nonzero direct edge
in Design A, and it means headway is the main channel but not the only one.

## A failed test, reported as failed

I tried a band-restricted IV as a falsification check (run 2SLS inside a narrow
headway band, expecting ~0 where no mechanism exists). It returned −205 s/min at
< 5 min and +0.60 s/min at 20–40 min. **The design is invalid**: restricting to a
headway band selects on the endogenous variable itself, collapsing the first stage
and exploding the ratio. The 20–40 min value near zero is *not* evidence of
anything — it comes from the same broken setup. Only the full-sample IV estimate
stands.

## What can be claimed

1. **Headway causally affects delay gained.** Stable under PC and FCI, survives the
   negative control, and is quantified by a strong instrument: ≈1.8 s per minute.
2. **The leader's delay acts mainly *through* headway**, not as a direct effect —
   though mediation is partial, with a residual direct association.
3. **Measured confounders do not explain the coupling.** Weather, hour and month
   move the partial correlations essentially not at all.
4. **Unmeasured confounding is not excluded.** FCI's reluctance to keep the direct
   edge is itself a hint that something latent sits behind it. This is exactly the
   gap the SUMO oracle is meant to close — in simulation we can *set* headway
   instead of instrumenting it.
