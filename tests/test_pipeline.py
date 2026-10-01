"""Test suite.

Focus is on the things that silently break a forecasting system: leakage in
lag features, metrics that lie on zero-demand series, safety stock that doesn't
respond to service level, and drift monitors that never fire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import KEYS, TARGET, build
from src.models import metrics as M
from src.monitoring.drift import psi, run_drift_check
from src.optimization.inventory import (
    critical_ratio, eoq, safety_stock_normal, safety_stock_quantile, simulate_policy, build_plan,
)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_perfect_forecast_is_zero_error():
    y = np.array([5, 10, 0, 3, 8], dtype=float)
    assert M.wape(y, y) == 0.0
    assert M.mae(y, y) == 0.0
    assert M.bias(y, y) == 0.0


def test_wape_handles_zero_demand_without_blowing_up():
    y = np.array([0, 0, 4, 6], dtype=float)
    p = np.array([1, 1, 4, 6], dtype=float)
    assert np.isfinite(M.wape(y, p))
    assert M.wape(y, p) == pytest.approx(0.2)


def test_bias_sign_is_interpretable():
    y = np.array([10.0, 10.0])
    assert M.bias(y, np.array([12.0, 12.0])) > 0   # over-forecast
    assert M.bias(y, np.array([8.0, 8.0])) < 0     # under-forecast


def test_pinball_penalises_asymmetrically():
    y = np.array([10.0])
    under = M.pinball_loss(y, np.array([8.0]), q=0.9)
    over = M.pinball_loss(y, np.array([12.0]), q=0.9)
    assert under > over, "a P90 forecast must punish under-forecasting harder"


def test_mase_beats_one_when_better_than_seasonal_naive():
    rng = np.random.default_rng(0)
    train = rng.normal(50, 10, 200)
    y = rng.normal(50, 10, 30)
    assert M.mase(y, np.full(30, 50.0), train, season=7) < 1.0


# --------------------------------------------------------------------------- #
# features -- leakage is the highest-cost bug in this repo
# --------------------------------------------------------------------------- #
def _toy_panel(n_days: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    for store in ["S01", "S02"]:
        for sku in ["SKU001", "SKU002"]:
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store_id": store,
                        "sku_id": sku,
                        "store_format": "Supermarket",
                        "region": "North",
                        "category": "Snacks",
                        "units_sold": np.arange(n_days) % 17 + 5,
                        "price": 2.0,
                        "base_price": 2.5,
                        "unit_cost": 1.5,
                        "promo_flag": 0,
                        "discount_pct": 0.0,
                        "holiday_flag": 0,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_lag_features_do_not_leak_the_target():
    df = build(_toy_panel())
    g = df[(df.store_id == "S01") & (df.sku_id == "SKU001")].sort_values("date")
    # lag_1 at time t must equal the target at t-1
    assert np.allclose(g["lag_1"].to_numpy()[1:], g[TARGET].to_numpy()[:-1])


def test_rolling_features_exclude_the_current_day():
    df = build(_toy_panel())
    g = df[(df.store_id == "S01") & (df.sku_id == "SKU001")].sort_values("date").reset_index(drop=True)
    row = g.iloc[40]
    prior = g[g["date"] < row["date"]].tail(7)[TARGET]
    assert row["roll_mean_7"] == pytest.approx(prior.mean(), rel=1e-6)


def test_features_never_mix_across_series():
    df = build(_toy_panel())
    first_rows = df.groupby(KEYS, observed=True).head(1)
    # after warm-up trimming every series starts with a valid, series-local lag
    assert first_rows["lag_28"].notna().all()


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
def test_safety_stock_increases_with_service_level():
    a = safety_stock_normal(sigma_daily=10, lead_time=7, service_level=0.90)
    b = safety_stock_normal(sigma_daily=10, lead_time=7, service_level=0.99)
    assert b > a


def test_safety_stock_increases_with_lead_time():
    a = safety_stock_normal(sigma_daily=10, lead_time=3, service_level=0.95)
    b = safety_stock_normal(sigma_daily=10, lead_time=14, service_level=0.95)
    assert b > a


def test_lead_time_variability_raises_safety_stock():
    stable = safety_stock_normal(10, 7, 0.95, sigma_lead_time=0.0, mean_daily=50)
    volatile = safety_stock_normal(10, 7, 0.95, sigma_lead_time=3.0, mean_daily=50)
    assert volatile > stable


def test_quantile_safety_stock_is_non_negative():
    assert safety_stock_quantile(p_high_lt=80, p50_lt=100) == 0.0
    assert safety_stock_quantile(p_high_lt=130, p50_lt=100) == 30.0


def test_critical_ratio_rises_with_stockout_cost():
    assert critical_ratio(underage_cost=9, overage_cost=1) > critical_ratio(2, 1)


def test_eoq_matches_closed_form():
    assert eoq(annual_demand=10_000, ordering_cost=50, unit_cost=4, holding_rate=0.25) == pytest.approx(
        np.sqrt(2 * 10_000 * 50 / (4 * 0.25))
    )


def test_eoq_is_zero_for_dead_stock():
    assert eoq(0, 50, 4, 0.25) == 0.0


def test_simulation_conserves_units():
    actuals = _toy_panel(30)[KEYS + ["date", "units_sold"]]
    fc = actuals.rename(columns={"units_sold": "pred_p50"}).copy()
    fc["pred_p90"] = fc["pred_p50"] * 1.3
    costs = actuals[KEYS].drop_duplicates().assign(unit_cost=1.5, price=2.0)
    plan = build_plan(fc, costs, service_level=0.95)
    sim = simulate_policy(actuals, plan)
    assert np.allclose(sim["units_sold"] + sim["lost_sales_units"], sim["demand"])
    assert (sim["fill_rate"] <= 1.0 + 1e-9).all()


def test_higher_service_level_yields_higher_fill_rate():
    actuals = _toy_panel(60)[KEYS + ["date", "units_sold"]]
    fc = actuals.rename(columns={"units_sold": "pred_p50"}).copy()
    fc["pred_p90"] = fc["pred_p50"] * 1.3
    costs = actuals[KEYS].drop_duplicates().assign(unit_cost=1.5, price=2.0)
    low = simulate_policy(actuals, build_plan(fc, costs, service_level=0.70))
    high = simulate_policy(actuals, build_plan(fc, costs, service_level=0.99))
    assert high["fill_rate"].mean() >= low["fill_rate"].mean()


# --------------------------------------------------------------------------- #
# drift
# --------------------------------------------------------------------------- #
def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.1


def test_psi_detects_a_shifted_distribution():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 5000), rng.normal(3, 1, 5000)
    assert psi(a, b) > 0.25


def test_drift_trigger_fires_on_injected_shock():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"price": rng.normal(10, 1, 3000), "lag_1": rng.normal(50, 8, 3000)})
    cur = ref.copy()
    shocked = pd.DataFrame({"price": rng.normal(18, 1, 3000), "lag_1": rng.normal(20, 8, 3000)})

    quiet = run_drift_check(ref, cur, ["price", "lag_1"])
    loud = run_drift_check(ref, shocked, ["price", "lag_1"])
    assert not quiet.retrain_required
    assert loud.retrain_required or (loud.feature_drift["band"] == "severe").sum() >= 2


# --------------------------------------------------------------------------- #
# API -- skipped automatically when no trained model is present
# --------------------------------------------------------------------------- #
from pathlib import Path  # noqa: E402

from src.config import CFG, resolve  # noqa: E402

MODEL_PRESENT = (resolve(CFG.api.model_dir) / "model_bundle.joblib").exists()
api = pytest.mark.skipif(not MODEL_PRESENT, reason="no trained model bundle")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as c:
        yield c


@api
def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"]
    assert body["n_features"] > 0


@api
def test_forecast_returns_the_requested_horizon(client):
    r = client.post("/forecast", json={"store_id": "S01", "sku_id": "SKU001", "horizon_days": 10})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 10
    assert all(p["p50"] >= 0 for p in body["forecast"])


@api
def test_p90_is_never_below_p50(client):
    r = client.post("/forecast", json={"store_id": "S01", "sku_id": "SKU001", "horizon_days": 14})
    assert all(p["p90"] >= p["p50"] for p in r.json()["forecast"])


@api
def test_promo_days_lift_the_forecast(client):
    base = client.post(
        "/forecast", json={"store_id": "S01", "sku_id": "SKU001", "horizon_days": 7}
    ).json()["total_p50"]
    promo = client.post(
        "/forecast",
        json={"store_id": "S01", "sku_id": "SKU001", "horizon_days": 7, "promo_days": [0, 1, 2, 3]},
    ).json()["total_p50"]
    assert promo > base


@api
def test_unknown_series_returns_404(client):
    r = client.post("/forecast", json={"store_id": "S99", "sku_id": "SKU999", "horizon_days": 7})
    assert r.status_code == 404


@api
def test_promo_days_outside_horizon_are_rejected(client):
    r = client.post(
        "/forecast", json={"store_id": "S01", "sku_id": "SKU001", "horizon_days": 5, "promo_days": [9]}
    )
    assert r.status_code == 422


@api
def test_low_stock_triggers_an_order(client):
    r = client.post(
        "/inventory/plan",
        json={"store_id": "S01", "sku_id": "SKU001", "horizon_days": 28, "on_hand": 0, "on_order": 0},
    ).json()
    assert r["should_order"] and r["recommended_order_qty"] > 0


@api
def test_ample_stock_suppresses_an_order(client):
    r = client.post(
        "/inventory/plan",
        json={"store_id": "S01", "sku_id": "SKU001", "horizon_days": 28,
              "on_hand": 100_000, "on_order": 0},
    ).json()
    assert not r["should_order"] and r["recommended_order_qty"] == 0


@api
def test_explain_returns_ranked_drivers(client):
    r = client.post("/explain", json={"store_id": "S01", "sku_id": "SKU001", "top_n": 5}).json()
    assert len(r["top_drivers"]) == 5
    shaps = [abs(d["shap"]) for d in r["top_drivers"]]
    assert shaps == sorted(shaps, reverse=True)
