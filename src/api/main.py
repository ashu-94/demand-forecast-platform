"""FastAPI serving layer.

Design notes
------------
* The model bundle and the recent history panel are loaded **once** at startup
  via the lifespan hook, not per request. Loading a 200-series panel per call
  would put ~2s of pandas work on every request.
* `/forecast` runs the same recursive roll-forward used in training, so the
  number served is the number that was evaluated. Train/serve skew in
  forecasting usually comes from evaluating one-step and serving multi-step.
* `/inventory/plan` is the endpoint a planning system actually calls: it takes
  on-hand and on-order and answers "order or not, and how much".
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import CFG, resolve
from src.api.schemas import (
    BatchForecastRequest, ExplainRequest, ForecastRequest, ForecastResponse,
    HealthResponse, InventoryRequest, InventoryResponse,
)
from src.models.forecast import recursive_forecast
from src.optimization.inventory import (
    critical_ratio, eoq, order_up_to_level, reorder_point, safety_stock_normal,
    safety_stock_quantile, z_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

STATE: dict = {}
KEYS = ["store_id", "sku_id"]
HISTORY_TAIL_DAYS = 120  # enough to satisfy the longest lag/rolling window


def _load() -> None:
    bundle_path = resolve(CFG.api.model_dir) / "model_bundle.joblib"
    if not bundle_path.exists():
        log.error("model bundle not found at %s -- run training first", bundle_path)
        return
    bundle = joblib.load(bundle_path)
    STATE["bundle"] = bundle
    STATE["models"] = bundle["models"]
    STATE["feature_cols"] = bundle["feature_cols"]
    STATE["quantiles"] = bundle["quantiles"]

    raw = pd.read_parquet(resolve(CFG.data.raw_path))
    raw["date"] = pd.to_datetime(raw["date"])
    cutoff = raw["date"].max() - pd.Timedelta(days=HISTORY_TAIL_DAYS)
    STATE["history"] = raw[raw["date"] >= cutoff].reset_index(drop=True)
    STATE["last_date"] = raw["date"].max()
    STATE["catalog"] = (
        raw.sort_values("date")
        .groupby(KEYS, observed=True)
        .last()
        .reset_index()[KEYS + ["store_format", "region", "category", "price",
                               "base_price", "unit_cost"]]
    )
    log.info("loaded model + %s history rows through %s", f"{len(STATE['history']):,}",
             STATE["last_date"].date())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    STATE.clear()


app = FastAPI(
    title="Demand Forecasting & Inventory Optimization API",
    description="Probabilistic demand forecasts and inventory policy recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require_model() -> None:
    if "models" not in STATE:
        raise HTTPException(503, "model not loaded; run `python -m src.models.train` first")


def _series_meta(store_id: str, sku_id: str) -> pd.Series:
    cat = STATE["catalog"]
    row = cat[(cat.store_id == store_id) & (cat.sku_id == sku_id)]
    if row.empty:
        raise HTTPException(404, f"unknown series {store_id}/{sku_id}")
    return row.iloc[0]


def _build_future_exog(req: ForecastRequest) -> pd.DataFrame:
    meta = _series_meta(req.store_id, req.sku_id)
    start = STATE["last_date"] + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=req.horizon_days, freq="D")
    base_price = float(meta["base_price"])
    price = float(req.planned_price or meta["price"])

    promo = np.zeros(req.horizon_days, dtype=int)
    promo[list(req.promo_days)] = 1
    eff_price = np.where(promo == 1, min(price, base_price * 0.85), price)

    md = dates.strftime("%m-%d")
    holiday = np.isin(md, ["01-26", "08-15", "12-25", "12-31", "01-01",
                           "03-08", "03-09", "10-24", "10-25", "11-12", "11-13"]).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "store_id": req.store_id,
            "sku_id": req.sku_id,
            "store_format": meta["store_format"],
            "region": meta["region"],
            "category": meta["category"],
            "price": eff_price.round(2),
            "base_price": base_price,
            "unit_cost": float(meta["unit_cost"]),
            "promo_flag": promo,
            "discount_pct": ((1 - eff_price / base_price) * 100).round(1),
            "holiday_flag": holiday,
        }
    )


def _forecast_series(req: ForecastRequest) -> pd.DataFrame:
    _require_model()
    hist = STATE["history"]
    series_hist = hist[(hist.store_id == req.store_id) & (hist.sku_id == req.sku_id)]
    if series_hist.empty:
        raise HTTPException(404, f"no history for {req.store_id}/{req.sku_id}")

    future = _build_future_exog(req)
    fc = recursive_forecast(
        STATE["models"], series_hist, future, STATE["feature_cols"]
    )
    return fc


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = "models" in STATE
    b = STATE.get("bundle", {})
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_trained_at=b.get("trained_at"),
        n_features=len(STATE["feature_cols"]) if loaded else None,
        quantiles=STATE.get("quantiles"),
        data_through=str(STATE["last_date"].date()) if loaded else None,
    )


@app.get("/metrics")
def metrics() -> dict:
    _require_model()
    return STATE["bundle"].get("metrics", {})


@app.get("/series")
def series(limit: int = 100) -> dict:
    _require_model()
    cat = STATE["catalog"]
    return {
        "n_series": len(cat),
        "stores": sorted(cat.store_id.unique().tolist()),
        "skus": sorted(cat.sku_id.unique().tolist()),
        "series": cat.head(limit)[KEYS + ["category", "region"]].to_dict("records"),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    fc = _forecast_series(req)
    points = [
        {
            "date": r["date"].date(),
            "p50": float(r["pred_p50"]),
            "p90": float(r.get("pred_p90", r["pred_p50"])),
            "promo": bool(r["promo_flag"]),
        }
        for _, r in fc.iterrows()
    ]
    return ForecastResponse(
        store_id=req.store_id,
        sku_id=req.sku_id,
        horizon_days=req.horizon_days,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_trained_at=STATE["bundle"]["trained_at"],
        total_p50=float(fc["pred_p50"].sum()),
        total_p90=float(fc.get("pred_p90", fc["pred_p50"]).sum()),
        forecast=points,
    )


@app.post("/forecast/batch")
def forecast_batch(req: BatchForecastRequest) -> dict:
    out, errors = [], []
    for item in req.items:
        try:
            out.append(forecast(item).model_dump())
        except HTTPException as e:
            errors.append({"store_id": item.store_id, "sku_id": item.sku_id, "error": e.detail})
    return {"n_ok": len(out), "n_failed": len(errors), "results": out, "errors": errors}


@app.post("/inventory/plan", response_model=InventoryResponse)
def inventory_plan(req: InventoryRequest) -> InventoryResponse:
    fc = _forecast_series(
        ForecastRequest(store_id=req.store_id, sku_id=req.sku_id, horizon_days=req.horizon_days)
    )
    meta = _series_meta(req.store_id, req.sku_id)
    inv = CFG.inventory
    lt = int(req.lead_time_days or inv.lead_time_days)
    rp = int(inv.review_period_days)

    unit_cost, price = float(meta["unit_cost"]), float(meta["price"])
    margin = max(price - unit_cost, 0.01)
    if req.service_level is not None:
        sl = float(req.service_level)
    else:
        cu = margin * float(inv.stockout_penalty_mult)
        co = unit_cost * float(inv.holding_cost_rate) * (lt / 365.0) + margin * 0.15
        sl = critical_ratio(cu, co)

    mean_daily = float(fc["pred_p50"].mean())
    sigma_daily = float(fc["pred_p50"].std(ddof=1) or 0.0)

    lt_win = fc.head(lt)
    ss_q = safety_stock_quantile(
        float(lt_win.get("pred_p90", lt_win["pred_p50"]).sum()), float(lt_win["pred_p50"].sum())
    )
    z_ref = z_score(0.90)
    ss_q *= z_score(sl) / z_ref if z_ref > 0 else 1.0
    ss_n = safety_stock_normal(sigma_daily, lt, sl, mean_daily=mean_daily)
    ss = float(np.ceil(max(ss_q, ss_n)))

    rop = float(np.ceil(reorder_point(mean_daily, lt, ss)))
    s_level = float(np.ceil(order_up_to_level(mean_daily, lt, rp, ss)))
    q_eoq = float(np.ceil(eoq(mean_daily * 365, float(inv.ordering_cost), unit_cost,
                              float(inv.holding_cost_rate))))

    position = req.on_hand + req.on_order
    should_order = position <= rop
    order_qty = float(np.ceil(max(s_level - position, 0))) if should_order else 0.0
    cover = position / mean_daily if mean_daily > 0 else float("inf")

    rationale = (
        f"Inventory position {position:.0f} {'is at or below' if should_order else 'is above'} "
        f"reorder point {rop:.0f} (lead-time demand {mean_daily * lt:.0f} + safety stock {ss:.0f}). "
        + (f"Order {order_qty:.0f} units to reach the order-up-to level of {s_level:.0f}."
           if should_order else "No order required this review cycle.")
    )

    return InventoryResponse(
        store_id=req.store_id, sku_id=req.sku_id,
        service_level=round(sl, 4), lead_time_days=lt,
        mean_daily_forecast=round(mean_daily, 2), safety_stock=ss,
        reorder_point=rop, order_up_to=s_level, eoq=q_eoq,
        inventory_position=position, should_order=should_order,
        recommended_order_qty=order_qty,
        days_of_cover=round(min(cover, 999.0), 1), rationale=rationale,
    )


@app.post("/explain")
def explain(req: ExplainRequest) -> dict:
    _require_model()
    from src.features.build_features import build
    from src.models.explain import ForecastExplainer

    hist = STATE["history"]
    s = hist[(hist.store_id == req.store_id) & (hist.sku_id == req.sku_id)]
    if s.empty:
        raise HTTPException(404, f"no history for {req.store_id}/{req.sku_id}")

    feats = build(s)
    if feats.empty:
        raise HTTPException(422, "insufficient history to build features")
    row = feats.tail(1)

    p50 = min(STATE["models"], key=lambda q: abs(q - 0.5))
    if "explainer" not in STATE:
        STATE["explainer"] = ForecastExplainer(STATE["models"][p50], STATE["feature_cols"])
    result = STATE["explainer"].explain_row(row, top_n=req.top_n)
    result.update({"store_id": req.store_id, "sku_id": req.sku_id,
                   "as_of": str(row["date"].iloc[0].date())})
    return result


@app.get("/monitoring/drift")
def drift() -> dict:
    path = resolve("artifacts") / "drift_report.json"
    if not path.exists():
        raise HTTPException(404, "no drift report; run `python -m scripts.run_monitoring`")
    import json

    return json.loads(path.read_text())


@app.post("/admin/reload")
def reload_model() -> dict:
    _load()
    return {"reloaded": True, "trained_at": STATE.get("bundle", {}).get("trained_at")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host=CFG.api.host, port=int(CFG.api.port), reload=False)
