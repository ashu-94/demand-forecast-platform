"""Feature engineering for a *global* forecasting model.

One model learns across all store x SKU series. Every lag/rolling feature is
computed inside the series group and shifted by at least one day, so nothing
leaks the target. Calendar + Fourier terms carry the seasonality that lags
alone cannot express at a 28-day horizon.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import CFG

log = logging.getLogger(__name__)

KEYS = ["store_id", "sku_id"]
TARGET = "units_sold"

CATEGORICALS = ["store_id", "sku_id", "store_format", "region", "category"]


def add_calendar(df: pd.DataFrame, fourier_order: int = 3) -> pd.DataFrame:
    d = df["date"]
    df["dayofweek"] = d.dt.dayofweek
    df["day"] = d.dt.day
    df["month"] = d.dt.month
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    # payday effect: 1st-5th and 25th-end of month
    df["is_payday_window"] = ((df["day"] <= 5) | (df["day"] >= 25)).astype(int)
    df["time_idx"] = (d - d.min()).dt.days

    doy = d.dt.dayofyear.to_numpy()
    for k in range(1, fourier_order + 1):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["discount_ratio"] = 1.0 - (df["price"] / df["base_price"])
    g = df.groupby(KEYS, observed=True)
    df["price_vs_28d_avg"] = df["price"] / g["price"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=7).mean()
    )
    # promo pressure over the coming week is known in advance (it's planned)
    df["promo_next_7d"] = g["promo_flag"].transform(
        lambda s: s.shift(-6).rolling(7, min_periods=1).sum()
    )
    df["days_since_promo"] = g["promo_flag"].transform(
        lambda s: s.groupby((s != 0).cumsum()).cumcount()
    )
    return df


def add_lags(df: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    g = df.groupby(KEYS, observed=True)[TARGET]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)

    base = g.shift(1)  # everything below is built off a shifted series -> no leakage
    tmp = df[KEYS].copy()
    tmp["_base"] = base
    gb = tmp.groupby(KEYS, observed=True)["_base"]
    for w in windows:
        df[f"roll_mean_{w}"] = gb.transform(lambda s, w=w: s.rolling(w, min_periods=2).mean())
        df[f"roll_std_{w}"] = gb.transform(lambda s, w=w: s.rolling(w, min_periods=2).std())
        df[f"roll_max_{w}"] = gb.transform(lambda s, w=w: s.rolling(w, min_periods=2).max())
    df["roll_zero_share_28"] = gb.transform(
        lambda s: s.eq(0).rolling(28, min_periods=7).mean()
    )
    # momentum + demand shape
    df["trend_7_28"] = df["roll_mean_7"] / df["roll_mean_28"].replace(0, np.nan)
    df["cv_28"] = df["roll_std_28"] / df["roll_mean_28"].replace(0, np.nan)
    df["dow_mean"] = df.groupby(KEYS + ["dayofweek"], observed=True)[TARGET].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    return df


def build(df: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True).copy()
    df = add_calendar(df, cfg.features.fourier_order)
    df = add_price_features(df)
    df = add_lags(df, list(cfg.features.lags), list(cfg.features.rolling_windows))

    for c in CATEGORICALS:
        df[c] = df[c].astype("category")

    # drop the burn-in period where long lags are undefined
    warmup = max(list(cfg.features.lags) + list(cfg.features.rolling_windows))
    before = len(df)
    df = df[df["lag_" + str(max(cfg.features.lags))].notna()].reset_index(drop=True)
    log.info("dropped %s warm-up rows (%sd)", f"{before - len(df):,}", warmup)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {TARGET, "date", "true_demand", "stockout_flag", "unit_cost", "base_price"}
    return [c for c in df.columns if c not in exclude]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from src.config import resolve

    raw = resolve(CFG.data.raw_path)
    df = pd.read_parquet(raw)
    out = build(df)
    dest = resolve(CFG.data.processed_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"{len(out):,} rows | {len(feature_columns(out))} features -> {dest}")


if __name__ == "__main__":
    main()
