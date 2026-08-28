# How the question changed — and why the path matters

This records the reasoning chain, not just the final answer. Each step was a
reasonable conclusion given what was known at the time, and each was revised by
the next piece of evidence. The revisions are part of the result.

---

## Step 1 — the starting assumption

*"Delay propagates through the junction; the job is to model how."*

Everything in Phase 0 was built on this. It is the framing the whole literature
uses (Dekker's diffusion, Li's GNN, Nguyen's GAT), so it was never questioned.

## Step 2 — the association, and a misleading summary

Measured `delay_gained = exit_delay − entry_delay` per train.
Median ≈ +7 s, mean ≈ +34 s, heavy right tail.

**Conclusion at the time:** *"the junction is neutral for the typical train; the
mean is driven by a tail."* Reported as: median ≈ 0, so model the distribution,
not the average.

**This was the weak link.** It is arithmetically true and analytically misleading.

## Step 3 — the challenge

*"The main hypothesis to test is whether the junction changes the delay at all —
and it doesn't seem to. Better to focus on the cases where it did change, and
understand why. I'd compare the delay distributions before and after."*

The point: we had been modelling *how much* delay propagates without first
establishing *whether* the junction does anything.

## Step 4 — testing it properly, and a correction

Compared entry and exit **distributions**, not per-train differences:

| quantile | entry | exit | shift |
|---|---|---|---|
| 5% | −6 s | −37 s | **−31** |
| 50% | 34 s | 63 s | **+29** |
| 90% | 217 s | 314 s | +97 |
| 99% | 585 s | 730 s | +145 |
| **sd** | **141 s** | **184 s** | **+42** |

Cohen's d = 0.318, KS = 0.148.

**The junction does change delay — substantially.** Step 2's summary was wrong
because *the median of the differences is not the difference of the medians*. The
typical train gains little, yet the distribution as a whole shifts +29 s at the
median.

And the mechanism is **dispersion, not translation**: the right tail lengthens
*and* the left tail does too (5th percentile −6 → −37 s). The junction **widens**
the distribution. Invisible if you look only at central tendency.

## Step 5 — the mass is in a minority

- **71.3%** of traversals: |gain| ≤ 60 s — effectively untouched
- **24.8%**: gain > 60 s
- that 24.8% accounts for **118%** of all delay the junction adds
  (over 100% because the 3.9% recovering >60 s offsets it)

So most of the continuous model's capacity was being spent on noise around zero.

## Step 6 — two hypotheses killed on the way

**Not mean reversion.** The obvious alternative was mechanical recovery into
schedule padding — late trains shed, early trains give back. Tested:
corr(entry delay, gain) = **+0.072**, *positive*. By decile of entry delay, mean
gain *rises* 26 → 50 s. **Delay begets delay.**

**A collider-bias error, caught.** Re-measuring the headway↔gain correlation
*inside* the moved group gave −0.067, weaker than in the unmoved group (−0.181).
That looks like a paradox but is an artifact: conditioning on the outcome and
measuring associations within that stratum is collider stratification bias. Same
family of error as the band-restricted IV in `CAUSAL.md`. The correct statistic is
the reverse conditioning — **P(moved | headway)** — which is clean.

## Step 7 — who gets moved

| | unmoved | moved | diff |
|---|---|---|---|
| headway | 491 s | **295 s** | −196 |
| leader's entry delay | 115 s | **233 s** | +118 |
| own entry delay | 75 s | 97 s | +22 |
| hour of day | 13.7 | 13.9 | +0.1 |
| temperature | 12.6 °C | 13.0 °C | +0.4 |

Not time of day, not weather: **short headway and a late leader.** Exactly the
knock-on mechanism the causal analysis identified — now visible as a
subpopulation rather than a slope.

Share moved by service: **P (peak reinforcements) 44.1%**, EURST 31.9%,
INT 28.3%, L 26.7%, IC 22.4%.

---

## The reformulated question

> Not *"how much delay does the junction add?"* but
> **"which trains does it move, and what makes them vulnerable?"**

`moved_model.py` implements it: **P(gain > threshold | headway, leader state,
service, context)**.

### It is a much better-posed problem

| | continuous `delay_gained` | binary `moved` |
|---|---|---|
| best model vs baseline | +4.0% MAE over persistence, loses in 1 of 9 folds | **ROC AUC 0.811**, PR AUC 0.639 |
| vs trivial rule | — | rule (headway<5min) AUC 0.692 → GBDT **0.811** |
| top-decile lift | — | **3.08×** base rate |
| calibration | — | predicted 0.04→0.81, actual 0.05→0.77 |

At threshold 120 s: AUC **0.828**, lift **3.93×**.

The continuous target buried a real signal under 71% of near-zero rows.

### The causal estimate survives the reformulation

Same instrument (scheduled headway), now a linear probability model:

| outcome | OLS | 2SLS | 95% CI |
|---|---|---|---|
| P(gain > 60 s) | −1.92 pp/min | **−0.91 pp/min** | [−0.99, −0.82] |
| P(gain > 120 s) | −1.23 pp/min | **−0.42 pp/min** | [−0.50, −0.37] |

One extra minute of headway causally removes ~0.9 percentage points of risk. OLS
again inflated ~2×, in the same direction as before.

### And the dose-response sharpens the literature

| headway | P(gain > 60 s) | P(gain > 120 s) |
|---|---|---|
| < 2 min | 60.1% | 38.6% |
| 2–3 min | 60.4% | 36.0% |
| 3–4 min | 36.3% | 20.2% |
| 4–5 min | 24.0% | 12.5% |
| 5–7 min | 16.7% | 7.8% |
| 7–10 min | 10.1% | 3.7% |
| 10–15 min | 9.6% | 2.8% |
| 15–20 min | 10.5% | 2.7% |
| 20–40 min | 10.6% | 3.1% |
| 40+ min | 11.2% | 5.4% |

**Risk collapses from 60% to 10% between 2 and 10 minutes, then goes flat.**

Two things follow. The effect **saturates at ~10 minutes, not the ~20 reported by
Li et al. 2024** — beyond 10 minutes extra headway buys nothing. And the plateau at
≈10% is an **irreducible floor**: trains moved for reasons unrelated to the train
ahead. That floor is the ceiling on what any headway-based intervention can achieve.

---

## Consequences for the rest of the programme

- **SUMO's calibration target changes.** It must reproduce the *width* of the
  delay distribution and the *fraction moved per headway band* — not the mean gain.
  A simulator matching the mean while missing the dispersion would pass the old
  test and fail the real one.
- **Operationally**, "this train risks losing over a minute" is actionable;
  "expect +7.3 s" is not.
- **The 10-minute saturation is a design number**: below it, headway is worth
  buying; above it, it is not.

---

## Step 8 — does the effect cascade beyond one train?

Everything above concerns one leader/follower pair. The natural next question:
**does a delayed train trigger a chain, or does the effect die after one hop?**

### Defining a chain, precisely

Three ingredients:

1. **Scope.** Only trains sharing `(date, direction, tunnel_track)` — the same
   physical resource, sorted by entry time.
2. **Link.** Train *i* links to train *i−1* iff **both** were moved (gain > 60 s)
   **and** the headway between them is ≤ 5 min.
3. **Chain.** Walk the sequence; every break in the link condition starts a new
   chain. Chain *position* of a moved train = 1 if not linked to a moved
   predecessor (a fresh incident), else predecessor's position + 1.

**Known brittleness.** The link uses a hard threshold on both ends. If train A is
badly delayed, B inherits some of it but lands at 55 s (just under the line), and C
inherits from B and crosses 60 s — this definition sees two length-1 chains, not a
3-link cascade, because B breaks the threshold. The chain counts below are
therefore a **lower bound** on real cascading.

### Cascades are real

Of 89,463 moved events (full year, threshold 60 s):

- **62.4%** are isolated — no continuation
- **37.6%** are chained to at least one more
- **63.2% of all moved trains** sit inside a chain of 2+
- chains up to length 18 observed

Persistence by headway to the next train:

| headway to next | P(next moved \| this moved) | P(next moved \| this NOT moved) | ratio |
|---|---|---|---|
| < 3 min | 92.1% | 49.0% | 1.9× |
| 3–5 min | 71.9% | 14.6% | **4.9×** |
| 5–10 min | 28.5% | 8.5% | 3.4× |
| > 10 min | 14.0% | 9.1% | 1.5× |

### The domino model

Target: at the moment train *i* exits the junction, will the *next* train on the
same track also be moved? Two feature sets:

- **retrospective** — uses the actual gap to the next train (only known once that
  train has departed; useful for post-hoc explanation)
- **operational** — uses only the *scheduled* gap (known from the timetable in
  advance; deployable in real time)

Trained on months 1–9, tested on 10–12 (held out, never seen in training):

| model | ROC AUC | PR AUC | lift @ top 10% |
|---|---|---|---|
| naive rule ("I'm late, so is next") | 0.689 | 0.397 | — |
| retrospective GBDT | 0.864 | 0.777 | 3.69× |
| **operational GBDT** | **0.859** | 0.738 | **3.47×** |

The strongest result in the whole analysis, and the only one directly usable as a
live dispatching alert: *"if you don't act now, this will hit the next train too"* —
computable before the fact, since it only needs the timetable and the current
train's own state.

### Is this just detecting momentum?

Concern: a model might just learn "already 4 in a row → of course a 5th follows,"
which is autocorrelation, not real prediction of onset. Tested by scoring the same
fitted model separately on subgroups defined by the **current train's** chain
position:

| current train's state | n | P(domino) | AUC |
|---|---|---|---|
| all moved trains | 21,673 | 51.7% | 0.872 |
| **fresh incident** (position 1, no history) | 13,675 | 47.0% | **0.855** |
| mid-cascade (position ≥ 2) | 7,998 | 59.6% | 0.895 |
| mid-cascade (position ≥ 3) | 3,215 | 64.5% | 0.905 |
| **currently on time** (position 0 — no "I'm late" signal at all) | 59,380 | 15.4% | **0.787** |

AUC does creep up with chain length (0.855 → 0.905), so momentum contributes
something — but the fresh-incident number (0.855) is barely below the overall
model, and even trains that are **currently on time** still get AUC 0.787 from
headway and context alone. **The model is not primarily riding autocorrelation.**
It detects structural risk — a short scheduled gap, a busy hour, a dense
track — largely independent of whether the current train happens to be late.

### Is 60 s an arbitrary threshold?

Yes, as originally chosen — no principled justification, just a round number. Retested
across 15 s to 300 s, retraining the domino model fresh at each threshold:

| threshold | base rate | chained share | domino AUC |
|---|---|---|---|
| 15 s | 45.4% | 57.8% | 0.791 |
| 30 s | 37.5% | 58.0% | 0.808 |
| 60 s | 26.4% | 58.7% | 0.838 |
| 90 s | 19.0% | 59.2% | 0.862 |
| 120 s | 13.9% | 59.4% | 0.878 |
| 180 s | 7.7% | 59.7% | 0.905 |
| 300 s | 2.6% | 58.7% | 0.932 |

The **chained share is flat at ~58–60% across the entire range** — cascading is not
an artifact of where the line was drawn. And AUC **rises monotonically** with the
threshold rather than degrading, because more extreme delay events are *more*
structurally determined, not less. 60 s is a reasonable operating point (base rate
26%, comparable to the earlier binary-moved analysis) but the qualitative finding
holds at every threshold tested.

## What this adds to the programme

- **Cascades, not just pairs, are the right unit for SUMO to reproduce**: chain-length
  distribution and the headway-conditioned persistence table, not only the
  single-hop knock-on rate.
- **The operational domino model is the one result in this whole analysis that is
  directly deployable** as a real-time alert, since it needs only the timetable and
  the current train's own (already-known) state.
- **The threshold used throughout (60 s) is arbitrary but not load-bearing** — every
  qualitative conclusion in this document holds from 15 s to 300 s.
