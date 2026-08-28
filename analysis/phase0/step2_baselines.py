"""Steps 2-4 - prediction task, temporal split, baselines, and a feature model.

Task: at the moment a train ENTERS the junction, predict its arrival delay at
the far side (`exit_delay`, seconds).

Split: temporal. Train on days 1-24, test on days 25-31. No shuffling, so no
leakage across time.

Baselines (a model earns its keep only by beating these, per Nguyen et al. 2025
where a persistence baseline beat GATv2 on MAE and RMSE):
  zero        - predict 0 (train is on time)
  persistence - predict exit_delay = entry_delay (junction changes nothing)
  mean-gain   - persistence plus the mean delay gained, learned on train days

Feature model: gradient-boosted trees over features knowable at entry only.
Fitted twice, with and without the headway/leader block, to isolate what the
train-interaction features are actually worth.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from common import OUT, report, scores, split

FEATURES_BASE = ["entry_delay", "hour", "tunnel_track", "direction", "service"]
FEATURES_INTER = ["headway_s", "leader_entry_delay"]
CATS = ["tunnel_track", "direction", "service"]
TARGET = "follower_exit_delay"


def prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in CATS:
        if c in X:
            X[c] = X[c].astype("category")
    return X


def fit_predict(tr, te, cols):
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6,
        categorical_features=[c for c in cols if c in CATS],
        random_state=0,
    )
    m.fit(prep(tr, cols), tr[TARGET])
    return m.predict(prep(te, cols)), m


def main() -> None:
    pairs = pd.read_csv(OUT / "pairs_202501.csv")
    pairs["day"] = pairs["date"].str[:2].astype(int)
    pairs = pairs.dropna(subset=[TARGET, "follower_entry_delay", "headway_s"])
    pairs = pairs.rename(columns={"follower_entry_delay": "entry_delay"})

    tr, te = split(pairs)
    print(f"train {len(tr):,} rows (days 1-24) | test {len(te):,} rows (days 25-31)")
    print(f"target: {TARGET}  mean={pairs[TARGET].mean():.1f}s median={pairs[TARGET].median():.1f}s")

    mean_gain = (tr[TARGET] - tr["entry_delay"]).mean()
    print(f"mean delay gained on train days: {mean_gain:+.1f}s")

    res = {
        "zero": scores(te[TARGET], [0] * len(te)),
        "persistence": scores(te[TARGET], te["entry_delay"]),
        "mean-gain": scores(te[TARGET], te["entry_delay"] + mean_gain),
    }

    pred_base, _ = fit_predict(tr, te, FEATURES_BASE)
    res["GBDT (no interaction)"] = scores(te[TARGET], pred_base)

    cols_full = FEATURES_BASE + FEATURES_INTER
    pred_full, model = fit_predict(tr, te, cols_full)
    res["GBDT (+ headway/leader)"] = scores(te[TARGET], pred_full)

    print("\n" + report(res))

    best = min(res, key=lambda k: res[k]["MAE"])
    pmae = res["persistence"]["MAE"]
    print(f"\nbest MAE: {best}")
    for k in ("GBDT (no interaction)", "GBDT (+ headway/leader)"):
        d = (pmae - res[k]["MAE"]) / pmae * 100
        print(f"  {k:<26} vs persistence: {d:+.1f}% MAE")

    d = (res["GBDT (no interaction)"]["MAE"] - res["GBDT (+ headway/leader)"]["MAE"])
    print(f"\nheadway/leader block is worth {d:+.2f}s MAE "
          f"({d / res['GBDT (no interaction)']['MAE'] * 100:+.2f}%)")

    short = te["headway_s"] < 300
    print(f"\nsubgroup headway <5min (n={short.sum():,}):")
    sub = {
        "persistence": scores(te[TARGET][short], te["entry_delay"][short]),
        "GBDT (no interaction)": scores(te[TARGET][short], pred_base[short.values]),
        "GBDT (+ headway/leader)": scores(te[TARGET][short], pred_full[short.values]),
    }
    print(report(sub))

    pd.DataFrame(res).T.to_csv(OUT / "step2_baselines.csv")
    print(f"\n-> {OUT / 'step2_baselines.csv'}")


if __name__ == "__main__":
    main()
