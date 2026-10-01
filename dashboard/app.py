"""Streamlit dashboard.

Reads artifacts produced by the pipeline and, where the API is running, calls
it directly so the dashboard exercises the same code path production does
rather than re-implementing the maths in the UI layer.

Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, resolve  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")
ART = resolve("artifacts")

st.set_page_config(page_title="Demand & Inventory Platform", page_icon="📦", layout="wide")


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300)
def load_parquet(name: str) -> pd.DataFrame | None:
    p = ART / name
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=300)
def load_csv(name: str) -> pd.DataFrame | None:
    p = ART / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=300)
def load_json(name: str) -> dict | None:
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame | None:
    p = resolve(CFG.data.raw_path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def api_get(path: str):
    try:
        import httpx

        r = httpx.get(f"{API_URL}{path}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(path: str, payload: dict):
    try:
        import httpx

        r = httpx.post(f"{API_URL}{path}", json=payload, timeout=120)
        return r.json() if r.status_code == 200 else {"_error": r.text}
    except Exception as e:
        return {"_error": str(e)}


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
st.title("📦 Demand Forecasting & Inventory Optimization")

health = api_get("/health")
metrics_json = load_json("metrics.json")
forecast_df = load_parquet("test_forecast.parquet")
plan_df = load_parquet("inventory_plan.parquet")
frontier_df = load_csv("policy_frontier.csv")
history = load_history()

c1, c2, c3, c4 = st.columns(4)
head = metrics_json.get("lgbm_recursive_28d") if metrics_json else None
base = metrics_json.get("seasonal_naive") if metrics_json else None
c1.metric("28-day WAPE", f"{head['wape']:.1%}" if head else "—",
          delta=f"{(head['wape'] - base['wape']) / base['wape']:.1%} vs naive" if head and base else None,
          delta_color="inverse")
c2.metric("MASE", f"{head['mase']:.3f}" if head and "mase" in head else "—",
          help="<1 beats the seasonal-naive baseline")
c3.metric("P90 coverage", f"{head['coverage_p90']:.1%}" if head and "coverage_p90" in head else "—",
          help="Should sit near 90%. Below means safety stock is too thin.")
c4.metric("API", "online" if health else "offline")

if not metrics_json:
    st.warning("No artifacts found. Run `make pipeline` first.")
    st.stop()

tabs = st.tabs(["📈 Forecast", "📦 Inventory", "🎯 Performance", "🔍 Explainability", "⚠️ Drift"])


# --------------------------------------------------------------------------- #
# 1. forecast
# --------------------------------------------------------------------------- #
with tabs[0]:
    if forecast_df is None or history is None:
        st.info("Run training to generate forecasts.")
    else:
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        left, right = st.columns([1, 3])
        with left:
            store = st.selectbox("Store", sorted(forecast_df.store_id.unique()))
            sku = st.selectbox("SKU", sorted(forecast_df[forecast_df.store_id == store].sku_id.unique()))
            lookback = st.slider("History shown (days)", 28, 180, 90, step=7)

        sel = forecast_df[(forecast_df.store_id == store) & (forecast_df.sku_id == sku)].sort_values("date")
        hist = history[(history.store_id == store) & (history.sku_id == sku)].sort_values("date")
        hist = hist[hist.date < sel.date.min()].tail(lookback)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.date, y=hist.units_sold, name="actual (history)",
                                 line=dict(color="#64748b", width=1.5)))
        fig.add_trace(go.Scatter(x=sel.date, y=sel.units_sold, name="actual (hold-out)",
                                 line=dict(color="#0f172a", width=2)))
        fig.add_trace(go.Scatter(x=sel.date, y=sel.pred_p90, name="P90",
                                 line=dict(color="#f97316", width=0, dash="dot"),
                                 fill=None, showlegend=True))
        fig.add_trace(go.Scatter(x=sel.date, y=sel.pred_p50, name="P50 forecast",
                                 line=dict(color="#2563eb", width=2.5),
                                 fill="tonexty", fillcolor="rgba(249,115,22,0.12)"))
        promo_days = sel[sel.promo_flag == 1]
        if not promo_days.empty:
            fig.add_trace(go.Scatter(x=promo_days.date, y=promo_days.units_sold, mode="markers",
                                     name="promo day", marker=dict(color="#dc2626", size=8, symbol="star")))
        fig.update_layout(height=460, hovermode="x unified", margin=dict(t=30, b=10),
                          yaxis_title="units", legend=dict(orientation="h", y=1.1))
        with right:
            st.plotly_chart(fig, width="stretch")

        from src.models.metrics import bias, wape

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("WAPE (this series)", f"{wape(sel.units_sold, sel.pred_p50):.1%}")
        m2.metric("Bias", f"{bias(sel.units_sold, sel.pred_p50):+.1%}",
                  help="Positive = chronic over-forecast")
        m3.metric("Actual total", f"{sel.units_sold.sum():,.0f}")
        m4.metric("Forecast total", f"{sel.pred_p50.sum():,.0f}")

        st.divider()
        st.subheader("Live forecast from the API")
        st.caption("Calls the running service, so this is the exact code path production uses.")
        a, b, c = st.columns([1, 1, 2])
        horizon = a.number_input("Horizon (days)", 7, 90, 14)
        promo_input = b.text_input("Promo day offsets", "", help="comma-separated, e.g. 2,3,4")
        if c.button("Request forecast", type="primary"):
            promo_days_list = [int(x) for x in promo_input.split(",") if x.strip().isdigit()]
            res = api_post("/forecast", {"store_id": store, "sku_id": sku,
                                         "horizon_days": int(horizon), "promo_days": promo_days_list})
            if res and "_error" not in res:
                fc = pd.DataFrame(res["forecast"])
                st.metric("Total P50 over horizon", f"{res['total_p50']:,.0f}",
                          delta=f"P90: {res['total_p90']:,.0f}")
                st.plotly_chart(px.line(fc, x="date", y=["p50", "p90"], markers=True),
                                width="stretch")
            else:
                st.error(f"API unavailable at {API_URL}. Start it with `make api`.")


# --------------------------------------------------------------------------- #
# 2. inventory
# --------------------------------------------------------------------------- #
with tabs[1]:
    if frontier_df is None:
        st.info("Run `python -m scripts.run_optimization`.")
    else:
        st.subheader("Efficient frontier: inventory investment vs service achieved")
        st.caption(
            "Both policies are swept across the same service-level targets. Comparing two "
            "policies at their own settings is meaningless — it only rewards whoever holds "
            "more stock. The curve that sits lower and further right is strictly better."
        )
        fig = px.line(frontier_df, x="avg_inventory_value", y="fill_rate", color="policy",
                      markers=True, hover_data=["target_service_level", "total_cost"],
                      labels={"avg_inventory_value": "average inventory value",
                              "fill_rate": "fill rate achieved"})
        fig.update_layout(height=430, yaxis_tickformat=".1%",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

        naive = frontier_df[frontier_df.policy == "naive_ma28"]
        ml = frontier_df[frontier_df.policy == "ml_quantile"]
        import numpy as np

        rows = []
        lo = max(naive.fill_rate.min(), ml.fill_rate.min())
        hi = min(naive.fill_rate.max(), ml.fill_rate.max())
        for t in [x for x in (0.90, 0.93, 0.95, 0.97) if lo <= x <= hi]:
            n = np.interp(t, naive.sort_values("fill_rate").fill_rate,
                          naive.sort_values("fill_rate").avg_inventory_value)
            m = np.interp(t, ml.sort_values("fill_rate").fill_rate,
                          ml.sort_values("fill_rate").avg_inventory_value)
            rows.append({"target fill rate": f"{t:.0%}", "naive inventory": round(n),
                         "ML inventory": round(m), "reduction": round(n - m),
                         "reduction %": f"{(n - m) / n:.2%}"})
        if rows:
            st.markdown("**Working capital required at matched fill rate**")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if plan_df is not None:
            st.divider()
            st.subheader("Reorder plan")
            f1, f2 = st.columns(2)
            stores = f1.multiselect("Stores", sorted(plan_df.store_id.unique()),
                                    default=sorted(plan_df.store_id.unique())[:3])
            top_n = f2.slider("Rows", 10, 200, 25)
            view = plan_df[plan_df.store_id.isin(stores)] if stores else plan_df
            cols = ["store_id", "sku_id", "mean_daily", "sigma_daily", "service_level",
                    "safety_stock", "reorder_point", "order_up_to", "eoq", "holding_cost_annual"]
            st.dataframe(view.sort_values("mean_daily", ascending=False).head(top_n)[cols].round(2),
                         width="stretch", hide_index=True)

            st.download_button("Download full plan (CSV)", plan_df.to_csv(index=False),
                               "inventory_plan.csv", "text/csv")

        st.divider()
        st.subheader("Live order recommendation")
        q1, q2, q3, q4 = st.columns(4)
        pstore = q1.selectbox("Store ", sorted(plan_df.store_id.unique()) if plan_df is not None else [])
        psku = q2.selectbox("SKU ", sorted(plan_df.sku_id.unique()) if plan_df is not None else [])
        on_hand = q3.number_input("On hand", 0, 1_000_000, 200)
        on_order = q4.number_input("On order", 0, 1_000_000, 0)
        if st.button("Get recommendation", type="primary"):
            res = api_post("/inventory/plan", {"store_id": pstore, "sku_id": psku,
                                               "on_hand": on_hand, "on_order": on_order})
            if res and "_error" not in res:
                if res["should_order"]:
                    st.error(f"**ORDER {res['recommended_order_qty']:,.0f} units**")
                else:
                    st.success("**No order required this cycle**")
                st.write(res["rationale"])
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Safety stock", f"{res['safety_stock']:,.0f}")
                d2.metric("Reorder point", f"{res['reorder_point']:,.0f}")
                d3.metric("Order-up-to", f"{res['order_up_to']:,.0f}")
                d4.metric("Days of cover", f"{res['days_of_cover']}")
            else:
                st.error(f"API unavailable at {API_URL}.")


# --------------------------------------------------------------------------- #
# 3. performance
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Model vs baselines (28-day hold-out)")
    summary = pd.DataFrame(metrics_json).T
    show = [c for c in ["wape", "mae", "rmse", "mase", "bias", "smape"] if c in summary.columns]
    st.dataframe(summary[show].round(4), width="stretch")

    st.caption(
        "`lgbm_one_step` is the optimistic number — it sees yesterday's actuals. "
        "`lgbm_recursive_28d` rolls the model forward without them and is the number "
        "that matches what the API serves."
    )

    fig = px.bar(summary.reset_index().rename(columns={"index": "model"}),
                 x="model", y="wape", color="model", text_auto=".3f")
    fig.update_layout(height=340, showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    cov_cols = [c for c in summary.columns if c.startswith("coverage")]
    if cov_cols:
        st.subheader("Quantile calibration")
        cal = summary[cov_cols].dropna(how="all")
        st.dataframe(cal.round(4), width="stretch")
        st.caption("Nominal targets are 50% and 90%. Large gaps mean the safety stock "
                   "derived from these quantiles will be systematically wrong.")

    per_series = load_csv("per_series_wape.csv")
    if per_series is not None:
        st.subheader("Worst series by WAPE (volume-weighted view)")
        fig = px.scatter(per_series, x="volume", y="wape", color="bias",
                         hover_data=["store_id", "sku_id"],
                         color_continuous_scale="RdBu_r", range_color=[-1, 1])
        fig.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.caption("High-volume series with high WAPE are where forecast work pays off first.")


# --------------------------------------------------------------------------- #
# 4. explainability
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Global feature importance (mean |SHAP|)")
    shap_imp = load_csv("shap_global_importance.csv")
    if shap_imp is not None:
        fig = px.bar(shap_imp.head(20).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
                     orientation="h")
        fig.update_layout(height=520, margin=dict(l=10))
        st.plotly_chart(fig, width="stretch")
    summary_png = ART / "shap_summary.png"
    if summary_png.exists():
        st.image(str(summary_png), caption="SHAP beeswarm — direction and magnitude per feature")

    st.divider()
    st.subheader("Explain a single forecast")
    e1, e2 = st.columns(2)
    estore = e1.selectbox("Store  ", sorted(plan_df.store_id.unique()) if plan_df is not None else [])
    esku = e2.selectbox("SKU  ", sorted(plan_df.sku_id.unique()) if plan_df is not None else [])
    if st.button("Explain"):
        res = api_post("/explain", {"store_id": estore, "sku_id": esku, "top_n": 8})
        if res and "_error" not in res:
            st.metric("Prediction", f"{res['prediction']:,.1f}",
                      delta=f"base {res['base_value']:,.1f}")
            drivers = pd.DataFrame(res["top_drivers"])
            fig = px.bar(drivers.sort_values("shap"), x="shap", y="feature", orientation="h",
                         color="shap", color_continuous_scale="RdBu", text="value")
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.error(f"API unavailable at {API_URL}.")


# --------------------------------------------------------------------------- #
# 5. drift
# --------------------------------------------------------------------------- #
with tabs[4]:
    drift_json = load_json("drift_report.json")
    if drift_json is None:
        st.info("Run `python -m scripts.run_monitoring`.")
    else:
        st.caption(
            "The monitor is validated against an injected shock as well as live data. "
            "A monitor that never fires is indistinguishable from a broken one."
        )
        for scenario, label in [("natural", "Natural drift (last 28d vs prior 90d)"),
                                ("injected_shock", "Injected supply shock (control)")]:
            block = drift_json.get(scenario)
            if not block:
                continue
            st.subheader(label)
            if block["retrain_required"]:
                st.error("Retrain triggered: " + "; ".join(block["triggers"]))
            else:
                st.success("Stable — no retrain required")
            k1, k2 = st.columns(2)
            pd_block = block.get("prediction_drift") or {}
            k1.metric("Prediction PSI", f"{pd_block.get('psi', 0):.4f}",
                      delta=pd_block.get("band"))
            perf = block.get("performance") or {}
            if "relative_change" in perf:
                k2.metric("WAPE vs baseline", f"{perf.get('current_wape', 0):.4f}",
                          delta=f"{perf['relative_change']:+.1%}", delta_color="inverse")

            csv_name = "drift_natural.csv" if scenario == "natural" else "drift_injected.csv"
            table = load_csv(csv_name)
            if table is not None:
                st.dataframe(table.head(12), width="stretch", hide_index=True)
            st.divider()

        st.info(
            "PSI is an effect size; the KS test is a significance test. At 200k rows KS flags "
            "differences far too small to matter, which is why the retrain trigger keys on PSI "
            "and on realised WAPE rather than on p-values."
        )
