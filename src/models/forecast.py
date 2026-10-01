"""Recursive multi-step forecasting.

The model is trained one-step-ahead, but the business needs a 28-day forecast
to place a purchase order. So at inference we roll forward day by day: predict
day t, append the prediction to history, rebuild lag/rolling features, predict
day t+1, and so on.

This is the honest evaluation. Scoring a model that gets to see yesterday's
*actual* sales 28 days out is a leak that makes accuracy look ~2x better than
it is in production.

Future covariates (price, planned promo, holidays) are assumed known -- true in
retail, where promo calendars are locked weeks ahead.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.build_features import KEYS, TARGET, build

log = logging.getLogger(__name__)


def make_future_frame(history: pd.DataFrame, future_exog: pd.DataFrame) -> pd.DataFrame:
    """Stack known history with the future rows we need predictions for."""
    future = future_exog.copy()
    if TARGET not in future.columns:
        future[TARGET] = np.nan
    cols = [c for c in history.columns if c in future.columns or c == TARGET]
    return pd.concat([history[cols], future[cols]], ignore_index=True)


def recursive_forecast(
    models: dict[float, object],
    history: pd.DataFrame,
    future_exog: pd.DataFrame,
    feature_cols: list[str],
    round_output: bool = True,
) -> pd.DataFrame:
    """Roll a trained model forward across the forecast horizon.

    Parameters
    ----------
    models : {quantile: fitted LGBMRegressor}. Quantile 0.5 is the point forecast.
    history : raw (pre-feature) rows with actual `units_sold`.
    future_exog : raw future rows with date/keys/price/promo/holiday but no target.

    Returns the future rows with one column per quantile: `pred_p50`, `pred_p90`, ...
    """
    panel = make_future_frame(history, future_exog)
    panel["date"] = pd.to_datetime(panel["date"])
    future_dates = sorted(pd.to_datetime(future_exog["date"]).unique())

    p50_key = min(models, key=lambda x: abs(x - 0.5))
    collected: list[pd.DataFrame] = []

    for i, d in enumerate(future_dates):
        feats = build(panel)
        mask = feats["date"] == d
        if not mask.any():
            log.warning("no rows for %s, skipping", d)
            continue

        X = feats.loc[mask, feature_cols]
        step = feats.loc[mask, KEYS + ["date"]].reset_index(drop=True)

        for q, model in models.items():
            p = np.clip(model.predict(X), 0, None)
            step[f"pred_p{int(q * 100)}"] = np.round(p) if round_output else p

        # feed the point forecast back in as if it were an observation
        writeback = step[KEYS + ["date", f"pred_p{int(p50_key * 100)}"]].rename(
            columns={f"pred_p{int(p50_key * 100)}": "_pred"}
        )
        panel = panel.merge(writeback, on=KEYS + ["date"], how="left")
        fill = panel["_pred"].notna() & panel[TARGET].isna()
        panel.loc[fill, TARGET] = panel.loc[fill, "_pred"]
        panel = panel.drop(columns="_pred")

        collected.append(step)
        if (i + 1) % 7 == 0:
            log.info("  recursive step %s/%s", i + 1, len(future_dates))

    ref = pd.concat(collected, ignore_index=True)
    ref["date"] = pd.to_datetime(ref["date"])
    out = future_exog.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.merge(ref, on=KEYS + ["date"], how="left")
