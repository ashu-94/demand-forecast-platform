"""Turn a probabilistic demand forecast into inventory decisions.

A forecast that nobody orders against is a science project. This module closes
the loop:

    quantile forecast -> safety stock -> reorder point -> order quantity -> cost

Two ways to size safety stock are implemented, and they disagree in useful ways:

* `safety_stock_normal`  -- classic z * sigma_LT. Assumes normal demand. Fine for
  fast movers, badly wrong for intermittent SKUs where demand is skewed.
* `safety_stock_quantile` -- take it straight from the P90 forecast the model
  produced. Distribution-free, so it holds up on slow movers.

The newsvendor critical ratio is what actually sets the service level when you
know the cost of being short vs the cost of being long, instead of a service
level someone picked in a meeting.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from src.config import CFG


# --------------------------------------------------------------------------- #
# service level
# --------------------------------------------------------------------------- #
def critical_ratio(underage_cost: float, overage_cost: float) -> float:
    """Newsvendor optimal service level = Cu / (Cu + Co).

    underage (Cu) = lost margin + goodwill when you stock out
    overage  (Co) = holding + markdown/obsolescence when you over-order
    """
    total = underage_cost + overage_cost
    return float(np.clip(underage_cost / total, 0.5, 0.999)) if total > 0 else 0.95


def z_score(service_level: float) -> float:
    return float(stats.norm.ppf(np.clip(service_level, 0.5, 0.9999)))


# --------------------------------------------------------------------------- #
# safety stock
# --------------------------------------------------------------------------- #
def safety_stock_normal(
    sigma_daily: float, lead_time: float, service_level: float, sigma_lead_time: float = 0.0,
    mean_daily: float = 0.0,
) -> float:
    """z * sqrt(LT * sigma_d^2 + mu_d^2 * sigma_LT^2).

    The second term matters: variable lead times often dominate demand variance,
    which is why 'just raise the service level' fails to fix stockouts.
    """
    var = lead_time * sigma_daily**2 + (mean_daily**2) * (sigma_lead_time**2)
    return float(max(z_score(service_level) * np.sqrt(max(var, 0.0)), 0.0))


def safety_stock_quantile(
    p_high_lt: float, p50_lt: float
) -> float:
    """Difference between the high-quantile and median forecast over lead time.

    Distribution-free -- inherits whatever skew the model learned.
    """
    return float(max(p_high_lt - p50_lt, 0.0))


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
def reorder_point(mean_daily: float, lead_time: float, safety_stock: float) -> float:
    return float(mean_daily * lead_time + safety_stock)


def order_up_to_level(mean_daily: float, lead_time: float, review_period: float, ss: float) -> float:
    """Periodic-review (R,S) target. Covers lead time + the review gap."""
    return float(mean_daily * (lead_time + review_period) + ss)


def eoq(annual_demand: float, ordering_cost: float, unit_cost: float, holding_rate: float) -> float:
    """Economic order quantity: sqrt(2DS / H)."""
    h = unit_cost * holding_rate
    if h <= 0 or annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * ordering_cost / h))


@dataclass
class InventoryPlan:
    store_id: str
    sku_id: str
    mean_daily_forecast: float
    sigma_daily: float
    service_level: float
    safety_stock: float
    reorder_point: float
    order_up_to: float
    eoq: float
    expected_annual_demand: float
    unit_cost: float
    holding_cost_annual: float
    stockout_risk: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_plan(
    forecast: pd.DataFrame,
    cost_table: pd.DataFrame,
    cfg=CFG,
    p_high: int = 90,
    use_newsvendor: bool = True,
    service_level: float | None = None,
    quantile_ss: bool = True,
    sigma_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Produce one inventory plan row per store x SKU from a horizon forecast.

    `forecast` needs: store_id, sku_id, date, pred_p50, pred_p{p_high}
    `cost_table` needs: store_id, sku_id, unit_cost, price
    """
    inv = cfg.inventory
    lt, rp = float(inv.lead_time_days), float(inv.review_period_days)
    keys = ["store_id", "sku_id"]

    agg = (
        forecast.groupby(keys, observed=True)
        .agg(
            horizon_days=("date", "nunique"),
            total_p50=("pred_p50", "sum"),
            mean_daily=("pred_p50", "mean"),
            sigma_daily=("pred_p50", "std"),
        )
        .reset_index()
    )

    # lead-time-window aggregates from the quantile forecasts
    fc = forecast.sort_values(keys + ["date"])
    lt_win = (
        fc.groupby(keys, observed=True)
        .head(int(lt))
        .groupby(keys, observed=True)
        .agg(p50_lt=("pred_p50", "sum"), phigh_lt=(f"pred_p{p_high}", "sum"))
        .reset_index()
    )

    plan = agg.merge(lt_win, on=keys, how="left").merge(cost_table, on=keys, how="left")

    # Safety stock must be sized on forecast *error*, not on how much the
    # forecast itself moves. A flat forecast has zero variance and would
    # otherwise get zero safety stock -- while being the least reliable of all.
    if sigma_table is not None:
        plan = plan.drop(columns=["sigma_daily"]).merge(
            sigma_table[keys + ["sigma_daily"]], on=keys, how="left"
        )
    plan["sigma_daily"] = plan["sigma_daily"].fillna(0.0)

    if service_level is not None:
        plan["service_level"] = float(service_level)
    elif use_newsvendor:
        margin = (plan["price"] - plan["unit_cost"]).clip(lower=0.01)
        cu = margin * float(inv.stockout_penalty_mult)
        co = plan["unit_cost"] * float(inv.holding_cost_rate) * (lt / 365.0) + margin * 0.15
        plan["service_level"] = [critical_ratio(a, b) for a, b in zip(cu, co)]
    else:
        plan["service_level"] = float(inv.service_level)

    plan["ss_normal"] = [
        safety_stock_normal(s, lt, sl, mean_daily=m)
        for s, sl, m in zip(plan["sigma_daily"], plan["service_level"], plan["mean_daily"])
    ]
    # Rescale the quantile-implied safety stock from its nominal p_high to the
    # requested service level, so both policies can be compared at matched SL.
    z_ref = z_score(p_high / 100.0)
    plan["ss_quantile"] = [
        safety_stock_quantile(hi, med) * (z_score(sl) / z_ref if z_ref > 0 else 1.0)
        for hi, med, sl in zip(plan["phigh_lt"], plan["p50_lt"], plan["service_level"])
    ]
    plan["safety_stock"] = np.ceil(plan["ss_quantile"] if quantile_ss else plan["ss_normal"])

    plan["reorder_point"] = np.ceil(
        [reorder_point(m, lt, s) for m, s in zip(plan["mean_daily"], plan["safety_stock"])]
    )
    plan["order_up_to"] = np.ceil(
        [order_up_to_level(m, lt, rp, s) for m, s in zip(plan["mean_daily"], plan["safety_stock"])]
    )
    plan["expected_annual_demand"] = plan["mean_daily"] * 365.0
    plan["eoq"] = np.ceil(
        [
            eoq(d, float(inv.ordering_cost), c, float(inv.holding_cost_rate))
            for d, c in zip(plan["expected_annual_demand"], plan["unit_cost"])
        ]
    )
    plan["holding_cost_annual"] = (
        (plan["safety_stock"] + plan["eoq"] / 2.0) * plan["unit_cost"] * float(inv.holding_cost_rate)
    )
    plan["stockout_risk"] = (1.0 - plan["service_level"]).round(4)
    return plan


# --------------------------------------------------------------------------- #
# simulation -- does the policy actually pay off?
# --------------------------------------------------------------------------- #
def simulate_policy(
    actuals: pd.DataFrame,
    plan: pd.DataFrame,
    cfg=CFG,
    target_col: str = "units_sold",
) -> pd.DataFrame:
    """Replay actual demand against an (R,S) policy and cost it out.

    Returns per-series service level, fill rate, and total cost so you can
    compare a forecast-driven policy against a naive one on the same demand.
    """
    inv = cfg.inventory
    lt, rp = int(inv.lead_time_days), int(inv.review_period_days)
    keys = ["store_id", "sku_id"]
    plan_idx = plan.set_index(keys)

    rows = []
    for name, g in actuals.sort_values("date").groupby(keys, observed=True):
        if name not in plan_idx.index:
            continue
        p = plan_idx.loc[name]
        S, ROP = float(p["order_up_to"]), float(p["reorder_point"])
        unit_cost, price = float(p["unit_cost"]), float(p["price"])
        margin = max(price - unit_cost, 0.0)
        daily_hold = unit_cost * float(inv.holding_cost_rate) / 365.0

        on_hand = S
        pipeline: dict[int, float] = {}
        lost, sold, held, orders, stockout_days = 0.0, 0.0, 0.0, 0, 0
        onhand_sum = 0.0

        demand = g[target_col].to_numpy(dtype=float)
        for t, d in enumerate(demand):
            on_hand += pipeline.pop(t, 0.0)
            fulfilled = min(on_hand, d)
            on_hand -= fulfilled
            sold += fulfilled
            short = d - fulfilled
            lost += short
            if short > 0:
                stockout_days += 1
            held += on_hand * daily_hold
            onhand_sum += on_hand
            inv_pos = on_hand + sum(pipeline.values())
            if t % rp == 0 and inv_pos <= ROP:
                qty = max(S - inv_pos, 0.0)
                if qty > 0:
                    pipeline[t + lt] = pipeline.get(t + lt, 0.0) + qty
                    orders += 1

        total_demand = demand.sum()
        rows.append(
            {
                "store_id": name[0],
                "sku_id": name[1],
                "days": len(demand),
                "demand": total_demand,
                "units_sold": sold,
                "lost_sales_units": lost,
                "fill_rate": sold / total_demand if total_demand else 1.0,
                "cycle_service_level": 1 - stockout_days / len(demand),
                "n_orders": orders,
                "avg_on_hand_units": onhand_sum / len(demand),
                "avg_inventory_value": (onhand_sum / len(demand)) * unit_cost,
                "holding_cost": held,
                "ordering_cost": orders * float(inv.ordering_cost),
                "lost_margin": lost * margin,
                "total_cost": held + orders * float(inv.ordering_cost) + lost * margin,
                "revenue": sold * price,
            }
        )
    return pd.DataFrame(rows)
