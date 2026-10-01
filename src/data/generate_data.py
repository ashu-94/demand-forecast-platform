"""Generate a synthetic but realistic multi-store / multi-SKU sales history.

Demand is built from interpretable components so the model has real signal to
find and so evaluation numbers mean something:

    base level x trend x weekly seasonality x annual seasonality
      x promo lift x price elasticity x holiday lift  ->  Poisson draw

Stockouts are then applied, so `units_sold` is censored demand -- exactly the
problem you hit in production. `true_demand` is kept for diagnostics only and
is never fed to the model.

Swap this out for Kaggle "Store Item Demand", M5/Walmart, or Rossmann by
writing a loader that emits the same schema (see README).
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.config import CFG, resolve

log = logging.getLogger(__name__)

CATEGORIES = {
    "Beverages": dict(base=45, price=2.4, margin=0.30, elasticity=-1.6, seasonal="summer"),
    "Snacks": dict(base=38, price=1.8, margin=0.35, elasticity=-1.2, seasonal="flat"),
    "Dairy": dict(base=52, price=3.1, margin=0.22, elasticity=-0.9, seasonal="flat"),
    "Frozen": dict(base=22, price=5.4, margin=0.28, elasticity=-1.1, seasonal="winter"),
    "HouseholdCare": dict(base=15, price=6.8, margin=0.40, elasticity=-0.7, seasonal="flat"),
}

STORE_FORMATS = {"Hypermarket": 1.55, "Supermarket": 1.0, "Express": 0.55}
REGIONS = ["North", "South", "East", "West"]


def _holidays(index: pd.DatetimeIndex) -> pd.Series:
    """Indian retail calendar: Republic Day, Holi-ish, Independence, Diwali-ish, Christmas."""
    md = index.strftime("%m-%d")
    fixed = {"01-26", "08-15", "12-25", "12-31", "01-01"}
    festive = {"03-08", "03-09", "10-24", "10-25", "11-12", "11-13"}
    flag = pd.Series(0.0, index=index)
    flag[md.isin(fixed)] = 1.0
    flag[md.isin(festive)] = 1.0
    # pre-festival stock-up: 3 days before a festive date
    lift = flag.rolling(4, min_periods=1).max().shift(-3).fillna(0.0)
    return np.maximum(flag, 0.6 * lift)


def generate(
    n_stores: int,
    n_skus: int,
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    t = np.arange(n_days) / 365.25
    hol = _holidays(dates).to_numpy()

    # weekend uplift, Friday-Sunday heavy
    dow_mult = np.array([0.92, 0.88, 0.90, 0.97, 1.12, 1.35, 1.22])[dow]

    stores = []
    for s in range(n_stores):
        fmt = list(STORE_FORMATS)[s % len(STORE_FORMATS)]
        stores.append(
            dict(
                store_id=f"S{s + 1:02d}",
                store_format=fmt,
                region=REGIONS[s % len(REGIONS)],
                size_mult=STORE_FORMATS[fmt] * rng.uniform(0.85, 1.15),
            )
        )

    skus = []
    cat_names = list(CATEGORIES)
    for i in range(n_skus):
        cat = cat_names[i % len(cat_names)]
        meta = CATEGORIES[cat]
        skus.append(
            dict(
                sku_id=f"SKU{i + 1:03d}",
                category=cat,
                unit_price=round(meta["price"] * rng.uniform(0.75, 1.35), 2),
                unit_cost=None,
                margin_rate=meta["margin"],
                elasticity=meta["elasticity"],
                base=meta["base"] * rng.uniform(0.5, 1.6),
                seasonal=meta["seasonal"],
                # ~20% of SKUs are slow movers -> intermittent demand
                slow=rng.random() < 0.2,
            )
        )
    for s in skus:
        s["unit_cost"] = round(s["unit_price"] * (1 - s["margin_rate"]), 2)

    frames = []
    for store in stores:
        for sku in skus:
            level = sku["base"] * store["size_mult"]

            # --- seasonality -------------------------------------------------
            annual = 1.0 + 0.18 * np.sin(2 * np.pi * (doy - 80) / 365.25)
            if sku["seasonal"] == "summer":
                annual += 0.30 * np.sin(2 * np.pi * (doy - 110) / 365.25)
            elif sku["seasonal"] == "winter":
                annual += 0.30 * np.sin(2 * np.pi * (doy - 290) / 365.25)

            trend = 1.0 + rng.uniform(-0.06, 0.14) * t

            # --- price & promo ----------------------------------------------
            promo = (rng.random(n_days) < 0.10).astype(float)
            # promos cluster into 3-5 day runs
            promo = pd.Series(promo).rolling(4, min_periods=1).max().to_numpy()
            discount = promo * rng.choice([0.10, 0.15, 0.20, 0.30], size=n_days)
            price = sku["unit_price"] * (1 - discount)
            price_effect = (price / sku["unit_price"]) ** sku["elasticity"]

            holiday_effect = 1.0 + 0.55 * hol

            mu = level * trend * annual * dow_mult * price_effect * holiday_effect
            mu = np.clip(mu, 0.05, None)

            if sku["slow"]:
                mu *= 0.12
                units = rng.poisson(mu) * (rng.random(n_days) > 0.45)  # zero-inflated
            else:
                units = rng.negative_binomial(n=8, p=8 / (8 + mu))

            true_demand = units.astype(float)

            # --- supply-side censoring: stockouts ---------------------------
            oos = (rng.random(n_days) < 0.025)
            oos = pd.Series(oos).rolling(3, min_periods=1).max().astype(bool).to_numpy()
            sold = np.where(oos, np.floor(true_demand * rng.uniform(0, 0.4, n_days)), true_demand)

            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store_id": store["store_id"],
                        "sku_id": sku["sku_id"],
                        "store_format": store["store_format"],
                        "region": store["region"],
                        "category": sku["category"],
                        "units_sold": sold.astype(int),
                        "true_demand": true_demand.astype(int),
                        "price": price.round(2),
                        "base_price": sku["unit_price"],
                        "unit_cost": sku["unit_cost"],
                        "promo_flag": promo.astype(int),
                        "discount_pct": (discount * 100).round(1),
                        "holiday_flag": (hol > 0).astype(int),
                        "stockout_flag": oos.astype(int),
                    }
                )
            )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    log.info("generated %s rows | %s series", f"{len(df):,}", df.groupby(['store_id', 'sku_id']).ngroups)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=CFG.data.raw_path)
    args = ap.parse_args()

    df = generate(
        n_stores=CFG.data.n_stores,
        n_skus=CFG.data.n_skus,
        start=CFG.data.start_date,
        end=CFG.data.end_date,
        seed=CFG.project.seed,
    )
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df):,} rows -> {out}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
