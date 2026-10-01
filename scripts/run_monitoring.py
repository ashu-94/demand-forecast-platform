"""Run drift checks and demonstrate that the trigger actually fires.

Two scenarios are evaluated:
  A. natural  -- last 28 days vs the preceding 90. Should look stable.
  B. injected -- a simulated supply shock (prices +35%, promos halted, demand
     shifted). Should trip the retrain trigger. A monitor that never fires is
     indistinguishable from a monitor that is broken, so we prove it fires.

Run: python -m scripts.run_monitoring
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.config import CFG, resolve
from src.monitoring.drift import run_drift_check

log = logging.getLogger(__name__)

MONITOR_FEATURES = [
    "price", "discount_ratio", "promo_flag", "holiday_flag", "lag_1", "lag_7",
    "roll_mean_7", "roll_mean_28", "roll_std_28", "dow_mean", "trend_7_28",
    "cv_28", "category", "region", "store_format",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    bundle = joblib.load(resolve(CFG.api.model_dir) / "model_bundle.joblib")
    p50 = min(bundle["models"], key=lambda q: abs(q - 0.5))
    model, feature_cols = bundle["models"][p50], bundle["feature_cols"]
    baseline_wape = (
        bundle["metrics"].get("lgbm_recursive_28d", {}).get("wape")
        or bundle["metrics"]["lgbm_one_step"]["wape"]
    )

    df = pd.read_parquet(resolve(CFG.data.processed_path))
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=27)
    reference = df[(df["date"] < cutoff) & (df["date"] >= cutoff - pd.Timedelta(days=90))].copy()
    current = df[df["date"] >= cutoff].copy()

    for frame in (reference, current):
        frame["prediction"] = np.clip(model.predict(frame[feature_cols]), 0, None)

    results = {}

    print("=" * 88)
    print("SCENARIO A: natural drift (last 28d vs prior 90d)")
    print("=" * 88)
    rep_a = run_drift_check(
        reference, current, MONITOR_FEATURES,
        pred_col="prediction", target_col="units_sold", baseline_wape=baseline_wape,
    )
    print(rep_a.summary())
    print("\ntop feature drift:")
    print(rep_a.feature_drift.head(8).to_string(index=False))
    results["natural"] = {
        "retrain_required": rep_a.retrain_required,
        "triggers": rep_a.triggers,
        "performance": rep_a.performance,
        "prediction_drift": rep_a.prediction_drift,
    }

    # ---------------- injected shock -------------------------------------- #
    shocked = current.copy()
    shocked["price"] = shocked["price"] * 1.35
    shocked["discount_ratio"] = 1.0 - (shocked["price"] / shocked["base_price"])
    shocked["promo_flag"] = 0
    for c in ["lag_1", "lag_7", "roll_mean_7", "roll_mean_28", "dow_mean"]:
        shocked[c] = shocked[c] * 0.55
    shocked["units_sold"] = (shocked["units_sold"] * 0.55).round()
    shocked["prediction"] = np.clip(model.predict(shocked[feature_cols]), 0, None)

    print("\n" + "=" * 88)
    print("SCENARIO B: injected supply shock (+35% price, promos halted, demand -45%)")
    print("=" * 88)
    rep_b = run_drift_check(
        reference, shocked, MONITOR_FEATURES,
        pred_col="prediction", target_col="units_sold", baseline_wape=baseline_wape,
    )
    print(rep_b.summary())
    print("\ntop feature drift:")
    print(rep_b.feature_drift.head(8).to_string(index=False))
    results["injected_shock"] = {
        "retrain_required": rep_b.retrain_required,
        "triggers": rep_b.triggers,
        "performance": rep_b.performance,
        "prediction_drift": rep_b.prediction_drift,
    }

    out = resolve("artifacts")
    rep_a.feature_drift.to_csv(out / "drift_natural.csv", index=False)
    rep_b.feature_drift.to_csv(out / "drift_injected.csv", index=False)
    (out / "drift_report.json").write_text(json.dumps(results, indent=2, default=float))

    print("\n" + "=" * 88)
    ok = (not rep_a.retrain_required) and rep_b.retrain_required
    print(f"monitor validation: {'PASS' if ok else 'CHECK'} "
          f"(natural={rep_a.retrain_required}, injected={rep_b.retrain_required})")
    print(f"saved -> {out/'drift_report.json'}")

    # non-zero exit tells CI/CD to kick off a retrain job
    raise SystemExit(1 if rep_a.retrain_required else 0)


if __name__ == "__main__":
    main()
