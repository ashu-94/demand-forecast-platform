"""Train the global demand model and log everything to MLflow.

Run:  python -m src.models.train
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import CFG, resolve
from src.features.build_features import CATEGORICALS, KEYS, TARGET, feature_columns
from src.models import metrics as M
from src.models.forecast import recursive_forecast

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def time_split(df: pd.DataFrame, test_h: int, val_h: int):
    """Chronological split. No shuffling, ever -- that would leak the future."""
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_h - 1)
    val_start = test_start - pd.Timedelta(days=val_h)
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    log.info(
        "train %s (%s..%s) | val %s | test %s (%s..%s)",
        f"{len(train):,}", train['date'].min().date(), train['date'].max().date(),
        f"{len(val):,}", f"{len(test):,}",
        test['date'].min().date(), test['date'].max().date(),
    )
    return train, val, test


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def seasonal_naive(history: pd.DataFrame, future: pd.DataFrame, season: int = 7) -> np.ndarray:
    """Last same-weekday value carried forward. The bar every model must clear."""
    last = (
        history.sort_values("date")
        .groupby(KEYS + [history["date"].dt.dayofweek.rename("dow")], observed=True)[TARGET]
        .last()
        .rename("sn")
        .reset_index()
    )
    f = future.copy()
    f["dow"] = f["date"].dt.dayofweek
    merged = f.merge(last, on=KEYS + ["dow"], how="left")
    return merged["sn"].fillna(history[TARGET].mean()).to_numpy()


def moving_average(history: pd.DataFrame, future: pd.DataFrame, window: int = 28) -> np.ndarray:
    ma = (
        history.sort_values("date")
        .groupby(KEYS, observed=True)[TARGET]
        .apply(lambda s: s.tail(window).mean())
        .rename("ma")
        .reset_index()
    )
    return future.merge(ma, on=KEYS, how="left")["ma"].fillna(0).to_numpy()


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def fit_quantile_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    quantiles: list[float],
    params: dict,
) -> dict[float, lgb.LGBMRegressor]:
    models: dict[float, lgb.LGBMRegressor] = {}
    cats = [c for c in CATEGORICALS if c in feature_cols]
    for q in quantiles:
        log.info("fitting quantile %.2f", q)
        p = dict(params, alpha=q, random_state=CFG.project.seed)
        model = lgb.LGBMRegressor(**p)
        model.fit(
            train[feature_cols],
            train[TARGET],
            eval_set=[(val[feature_cols], val[TARGET])],
            eval_metric="quantile",
            categorical_feature=cats,
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        log.info("  best iter=%s", model.best_iteration_)
        models[q] = model
    return models


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-recursive", action="store_true", help="skip slow recursive eval")
    ap.add_argument("--register", action="store_true", help="register model in MLflow registry")
    args = ap.parse_args()

    import mlflow

    mlflow.set_tracking_uri(CFG.mlflow.tracking_uri)
    mlflow.set_experiment(CFG.mlflow.experiment_name)

    df = pd.read_parquet(resolve(CFG.data.processed_path))
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = feature_columns(df)
    quantiles = list(CFG.model.quantiles)

    train, val, test = time_split(df, CFG.split.test_horizon, CFG.split.val_horizon)

    with mlflow.start_run(run_name=f"lgbm-quantile-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.log_params(
            {
                "model": "LGBMRegressor-quantile",
                "n_features": len(feature_cols),
                "n_series": df.groupby(KEYS, observed=True).ngroups,
                "quantiles": str(quantiles),
                "test_horizon": CFG.split.test_horizon,
                **{f"lgbm_{k}": v for k, v in dict(CFG.model.params).items()},
            }
        )

        t0 = time.time()
        models = fit_quantile_models(train, val, feature_cols, quantiles, dict(CFG.model.params))
        mlflow.log_metric("train_seconds", round(time.time() - t0, 1))

        history = pd.concat([train, val], ignore_index=True)
        y_train_series = history.sort_values("date")[TARGET].to_numpy()

        # ---------------- baselines --------------------------------------- #
        results: dict[str, dict] = {}
        results["seasonal_naive"] = M.evaluate(
            test[TARGET], seasonal_naive(history, test), y_train_series
        )
        results["moving_avg_28"] = M.evaluate(
            test[TARGET], moving_average(history, test), y_train_series
        )

        # ---------------- one-step-ahead (optimistic upper bound) ---------- #
        one_step = {q: np.clip(m.predict(test[feature_cols]), 0, None) for q, m in models.items()}
        p50_key = min(models, key=lambda x: abs(x - 0.5))
        results["lgbm_one_step"] = M.evaluate(
            test[TARGET], one_step[p50_key], y_train_series, quantile_preds=one_step
        )

        # ---------------- recursive 28-day (the production number) --------- #
        if not args.no_recursive:
            log.info("running recursive %s-day forecast ...", CFG.split.test_horizon)
            exog_cols = [
                c for c in ["date", *KEYS, "store_format", "region", "category",
                            "price", "base_price", "unit_cost", "promo_flag",
                            "discount_pct", "holiday_flag"] if c in test.columns
            ]
            fc = recursive_forecast(
                models, history, test[exog_cols], feature_cols
            )
            merged = test[KEYS + ["date", TARGET]].merge(fc, on=KEYS + ["date"], how="left")
            qpreds = {q: merged[f"pred_p{int(q * 100)}"].to_numpy() for q in quantiles}
            results["lgbm_recursive_28d"] = M.evaluate(
                merged[TARGET], qpreds[p50_key], y_train_series, quantile_preds=qpreds
            )
            merged.to_parquet(resolve("artifacts/test_forecast.parquet"), index=False)
            mlflow.log_artifact(str(resolve("artifacts/test_forecast.parquet")))

            report = M.per_series_report(merged, TARGET, f"pred_p{int(p50_key * 100)}", KEYS)
            report.to_csv(resolve("artifacts/per_series_wape.csv"), index=False)
            mlflow.log_artifact(str(resolve("artifacts/per_series_wape.csv")))

        for name, res in results.items():
            for k, v in res.items():
                if v == v:  # skip NaN
                    mlflow.log_metric(f"{name}__{k}", v)

        # ---------------- feature importance ------------------------------ #
        imp = (
            pd.DataFrame(
                {"feature": feature_cols, "gain": models[p50_key].booster_.feature_importance("gain")}
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
        imp.to_csv(resolve("artifacts/feature_importance.csv"), index=False)
        mlflow.log_artifact(str(resolve("artifacts/feature_importance.csv")))

        # ---------------- persist ------------------------------------------ #
        art = resolve(CFG.api.model_dir)
        art.mkdir(parents=True, exist_ok=True)
        bundle = {
            "models": models,
            "feature_cols": feature_cols,
            "quantiles": quantiles,
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "train_end": str(history["date"].max().date()),
            "metrics": results,
        }
        joblib.dump(bundle, art / "model_bundle.joblib")
        (art / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

        mlflow.log_artifact(str(art / "model_bundle.joblib"))
        mlflow.log_artifact(str(art / "metrics.json"))
        if args.register:
            mlflow.sklearn.log_model(
                models[p50_key], name="model", registered_model_name="demand-forecaster-p50"
            )

        # ---------------- console summary ---------------------------------- #
        print("\n" + "=" * 78)
        print(f"MLflow run: {run.info.run_id}")
        print("=" * 78)
        summary = pd.DataFrame(results).T
        cols = [c for c in ["wape", "mae", "rmse", "mase", "bias", "smape"] if c in summary.columns]
        print(summary[cols].round(4).to_string())
        cov = [c for c in summary.columns if c.startswith(("coverage", "pinball"))]
        if cov:
            print("\nQuantile calibration:")
            print(summary[cov].dropna(how="all").round(4).to_string())
        print("\nTop 12 features by gain:")
        print(imp.head(12).to_string(index=False))
        print(f"\nsaved -> {art / 'model_bundle.joblib'}")


if __name__ == "__main__":
    main()
