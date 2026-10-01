"""Pydantic request/response contracts."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    store_id: str = Field(..., examples=["S01"])
    sku_id: str = Field(..., examples=["SKU001"])
    horizon_days: int = Field(14, ge=1, le=90)
    planned_price: float | None = Field(None, gt=0, description="Defaults to last known price")
    promo_days: list[int] = Field(
        default_factory=list, description="0-indexed offsets within the horizon that are on promo"
    )

    @field_validator("promo_days")
    @classmethod
    def _check_promo(cls, v: list[int], info):
        h = info.data.get("horizon_days", 14)
        bad = [d for d in v if d < 0 or d >= h]
        if bad:
            raise ValueError(f"promo_days outside horizon: {bad}")
        return v


class ForecastPoint(BaseModel):
    date: date
    p50: float
    p90: float
    promo: bool


class ForecastResponse(BaseModel):
    store_id: str
    sku_id: str
    horizon_days: int
    generated_at: str
    model_trained_at: str
    total_p50: float
    total_p90: float
    forecast: list[ForecastPoint]


class BatchForecastRequest(BaseModel):
    items: list[ForecastRequest] = Field(..., max_length=200)


class InventoryRequest(BaseModel):
    store_id: str
    sku_id: str
    horizon_days: int = Field(28, ge=7, le=90)
    on_hand: float = Field(0, ge=0)
    on_order: float = Field(0, ge=0)
    service_level: float | None = Field(None, gt=0.5, lt=1.0)
    lead_time_days: int | None = Field(None, ge=1, le=60)


class InventoryResponse(BaseModel):
    store_id: str
    sku_id: str
    service_level: float
    lead_time_days: int
    mean_daily_forecast: float
    safety_stock: float
    reorder_point: float
    order_up_to: float
    eoq: float
    inventory_position: float
    should_order: bool
    recommended_order_qty: float
    days_of_cover: float
    rationale: str


class ExplainRequest(BaseModel):
    store_id: str
    sku_id: str
    top_n: int = Field(6, ge=1, le=20)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_trained_at: str | None
    n_features: int | None
    quantiles: list[float] | None
    data_through: str | None
