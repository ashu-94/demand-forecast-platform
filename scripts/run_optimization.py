"""Quantify the business value of the forecast, honestly.

The naive way to A/B two inventory policies is to run each at its own settings
and compare total cost. That comparison is meaningless: whichever policy holds
more stock wins on service and loses on holding cost, and you can move the
result anywhere you like by tweaking one knob.

The correct comparison holds *service level constant* and asks how much
inventory each policy needs to get there. A better forecast means less stock
for the same fill rate. Sweeping the service level traces an efficient
frontier -- and the better policy's curve dominates everywhere.

Run: python -m scripts.run_optimization
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import CFG, resolve
from src.optimization.inventory import build_plan, simulate_policy

log = logging.getLogger(__name__)
KEYS = ["store_id", "sku_id"]
SERVICE_LEVELS = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995]


def build_naive_forecast(hist: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """28-day moving average + normal-assumption p90. The incumbent process."""
    g = hist.sort_values("date").groupby(KEYS, observed=True)["units_sold"]
    ma = g.apply(lambda s: s.tail(28).mean()).rename("pred_p50").reset_index()
    sd = g.apply(lambda s: s.tail(28).std()).rename("sd").reset_index()
    out = template[KEYS + ["date"]].merge(ma, on=KEYS, how="left").merge(sd, on=KEYS, how="left")
    out["pred_p90"] = out["pred_p50"] + 1.2816 * out["sd"].fillna(0)
    return out.drop(columns="sd")


def sweep(
    label: str,
    forecast: pd.DataFrame,
    cost_table: pd.DataFrame,
    actuals: pd.DataFrame,
    quantile_ss: bool,
    sigma_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for sl in SERVICE_LEVELS:
        plan = build_plan(
            forecast, cost_table, service_level=sl, quantile_ss=quantile_ss,
            sigma_table=sigma_table,
        )
        sim = simulate_policy(actuals, plan)
        rows.append(
            {
                "policy": label,
                "target_service_level": sl,
                "fill_rate": sim["units_sold"].sum() / sim["demand"].sum(),
                "cycle_service_level": sim["cycle_service_level"].mean(),
                "avg_inventory_value": sim["avg_inventory_value"].sum(),
                "lost_sales_units": sim["lost_sales_units"].sum(),
                "holding_cost": sim["holding_cost"].sum(),
                "lost_margin": sim["lost_margin"].sum(),
                "ordering_cost": sim["ordering_cost"].sum(),
                "total_cost": sim["total_cost"].sum(),
            }
        )
        log.info("%s @ SL=%.2f -> fill=%.4f inv=%.0f", label, sl, rows[-1]["fill_rate"],
                 rows[-1]["avg_inventory_value"])
    return pd.DataFrame(rows)


def inventory_at_target_fill(frontier: pd.DataFrame, target: float) -> float:
    """Interpolate the inventory a policy needs to hit a given fill rate."""
    f = frontier.sort_values("fill_rate")
    if target < f["fill_rate"].min() or target > f["fill_rate"].max():
        return float("nan")
    return float(np.interp(target, f["fill_rate"], f["avg_inventory_value"]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    fc = pd.read_parquet(resolve("artifacts/test_forecast.parquet"))
    fc["date"] = pd.to_datetime(fc["date"])
    raw = pd.read_parquet(resolve(CFG.data.raw_path))
    raw["date"] = pd.to_datetime(raw["date"])

    cost_table = (
        raw.sort_values("date")
        .groupby(KEYS, observed=True)[["unit_cost", "price"]]
        .last()
        .reset_index()
    )
    horizon_dates = fc["date"].unique()
    actuals = raw[raw["date"].isin(horizon_dates)][KEYS + ["date", "units_sold"]]
    hist = raw[raw["date"] < fc["date"].min()]

    naive_fc = build_naive_forecast(hist, fc)

    # The incumbent process sizes safety stock off recent demand volatility.
    hist_sigma = (
        hist.sort_values("date")
        .groupby(KEYS, observed=True)["units_sold"]
        .apply(lambda s: s.tail(56).std())
        .rename("sigma_daily")
        .reset_index()
    )

    frontier = pd.concat(
        [
            sweep("naive_ma28", naive_fc, cost_table, actuals, quantile_ss=False,
                  sigma_table=hist_sigma),
            sweep("ml_quantile", fc, cost_table, actuals, quantile_ss=True),
        ],
        ignore_index=True,
    )

    out_dir = resolve("artifacts")
    frontier.to_csv(out_dir / "policy_frontier.csv", index=False)

    # operational plan at the newsvendor-optimal service level
    plan_fc = build_plan(fc, cost_table, use_newsvendor=True)
    sim_fc = simulate_policy(actuals, plan_fc)
    plan_fc.to_parquet(out_dir / "inventory_plan.parquet", index=False)
    sim_fc.to_parquet(out_dir / "policy_sim_forecast.parquet", index=False)

    # ---------------- report ---------------------------------------------- #
    pd.set_option("display.width", 220)
    print("\n" + "=" * 104)
    print(f"EFFICIENT FRONTIER  |  {len(horizon_dates)} days x {len(plan_fc)} series")
    print("=" * 104)
    show = frontier.copy()
    show["fill_rate"] = (show["fill_rate"] * 100).round(2)
    show["cycle_service_level"] = (show["cycle_service_level"] * 100).round(2)
    for c in ["avg_inventory_value", "lost_sales_units", "holding_cost", "lost_margin", "total_cost"]:
        show[c] = show[c].round(0).astype(int)
    print(show[["policy", "target_service_level", "fill_rate", "cycle_service_level",
                "avg_inventory_value", "lost_sales_units", "holding_cost", "lost_margin",
                "total_cost"]].to_string(index=False))

    naive_f = frontier[frontier.policy == "naive_ma28"]
    ml_f = frontier[frontier.policy == "ml_quantile"]

    print("\n" + "-" * 104)
    print("INVENTORY REQUIRED AT MATCHED FILL RATE  (lower is better)")
    print("-" * 104)
    lo = max(naive_f["fill_rate"].min(), ml_f["fill_rate"].min())
    hi = min(naive_f["fill_rate"].max(), ml_f["fill_rate"].max())
    rows = []
    for target in [t for t in (0.90, 0.93, 0.95, 0.97, 0.98) if lo <= t <= hi]:
        n_inv = inventory_at_target_fill(naive_f, target)
        m_inv = inventory_at_target_fill(ml_f, target)
        rows.append(
            {
                "target_fill_rate": f"{target:.0%}",
                "naive_inventory": round(n_inv),
                "ml_inventory": round(m_inv),
                "reduction": round(n_inv - m_inv),
                "reduction_pct": round((n_inv - m_inv) / n_inv * 100, 2),
            }
        )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
        avg_red = np.mean([r["reduction_pct"] for r in rows])
        print(f"\naverage working-capital reduction at matched fill rate: {avg_red:.1f}%")
    else:
        print("no overlapping fill-rate range between policies")

    print("\n" + "-" * 104)
    print("OPERATIONAL PLAN @ newsvendor-optimal service level")
    print("-" * 104)
    print(f"fill rate           : {sim_fc['units_sold'].sum() / sim_fc['demand'].sum():.2%}")
    print(f"avg inventory value : {sim_fc['avg_inventory_value'].sum():,.0f}")
    print(f"total cost          : {sim_fc['total_cost'].sum():,.0f}")
    cols = ["store_id", "sku_id", "mean_daily", "sigma_daily", "service_level",
            "safety_stock", "reorder_point", "order_up_to", "eoq"]
    print("\ntop 8 series by volume:")
    print(plan_fc.sort_values("mean_daily", ascending=False).head(8)[cols].round(2).to_string(index=False))
    print(f"\nsaved -> {out_dir/'policy_frontier.csv'}, {out_dir/'inventory_plan.parquet'}")


if __name__ == "__main__":
    main()
