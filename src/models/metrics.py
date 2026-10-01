"""Forecast accuracy metrics.

MAPE is deliberately absent as a headline metric: retail demand has zeros and
near-zeros, where MAPE explodes or is undefined. WAPE (volume-weighted) and
MASE (scaled against a seasonal-naive baseline) are the metrics a supply-chain
team will actually accept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_arr(y_true) - _arr(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((_arr(y_true) - _arr(y_pred)) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y|. Robust to zeros."""
    y, p = _arr(y_true), _arr(y_pred)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / denom) if denom else float("nan")


def smape(y_true, y_pred) -> float:
    y, p = _arr(y_true), _arr(y_pred)
    denom = (np.abs(y) + np.abs(p)) / 2.0
    mask = denom > 0
    return float(np.mean(np.abs(y[mask] - p[mask]) / denom[mask])) if mask.any() else float("nan")


def bias(y_true, y_pred) -> float:
    """Positive = over-forecast. Chronic bias is what fills warehouses."""
    y, p = _arr(y_true), _arr(y_pred)
    denom = np.sum(np.abs(y))
    return float(np.sum(p - y) / denom) if denom else float("nan")


def mase(y_true, y_pred, y_train, season: int = 7) -> float:
    """Scaled by in-sample seasonal-naive MAE. <1 means we beat seasonal naive."""
    y, p, tr = _arr(y_true), _arr(y_pred), _arr(y_train)
    if len(tr) <= season:
        return float("nan")
    scale = np.mean(np.abs(tr[season:] - tr[:-season]))
    return float(np.mean(np.abs(y - p)) / scale) if scale > 0 else float("nan")


def pinball_loss(y_true, y_pred, q: float) -> float:
    """Quantile loss. The objective a P90 safety-stock forecast is judged on."""
    y, p = _arr(y_true), _arr(y_pred)
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y_true, y_pred_q, q: float) -> float:
    """Empirical share of actuals at or below the quantile forecast.

    Should land near `q`. Far below -> the safety stock is too thin.
    """
    y, p = _arr(y_true), _arr(y_pred_q)
    return float(np.mean(y <= p))


def evaluate(
    y_true,
    y_pred,
    y_train=None,
    season: int = 7,
    quantile_preds: dict[float, np.ndarray] | None = None,
) -> dict[str, float]:
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, season)
    for q, pred in (quantile_preds or {}).items():
        out[f"pinball_p{int(q * 100)}"] = pinball_loss(y_true, pred, q)
        out[f"coverage_p{int(q * 100)}"] = coverage(y_true, pred, q)
    return out


def per_series_report(df: pd.DataFrame, y_col: str, pred_col: str, keys: list[str]) -> pd.DataFrame:
    """WAPE and bias per store x SKU -- surfaces which series to fix first."""
    rows = []
    for name, g in df.groupby(keys, observed=True):
        rows.append(
            dict(
                zip(keys, name if isinstance(name, tuple) else (name,)),
                n=len(g),
                volume=float(g[y_col].sum()),
                wape=wape(g[y_col], g[pred_col]),
                bias=bias(g[y_col], g[pred_col]),
            )
        )
    return pd.DataFrame(rows).sort_values("volume", ascending=False).reset_index(drop=True)
