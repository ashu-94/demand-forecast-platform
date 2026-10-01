"""SHAP explainability.

Two audiences, two questions:

* Planners ask "why is this SKU's forecast up 30% next week?" -> per-row SHAP,
  served through the API so the answer sits next to the number.
* Data science / audit asks "what is the model keying on globally, and has that
  changed since the last retrain?" -> mean |SHAP| by feature, logged to MLflow.

TreeExplainer is exact for LightGBM and fast enough to run inside a request
for a single row.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap

log = logging.getLogger(__name__)


class ForecastExplainer:
    def __init__(self, model, feature_cols: list[str]):
        self.model = model
        self.feature_cols = feature_cols
        self.explainer = shap.TreeExplainer(model)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        vals = self.explainer.shap_values(X[self.feature_cols])
        return np.asarray(vals)

    def global_importance(self, X: pd.DataFrame, sample: int = 5000) -> pd.DataFrame:
        if len(X) > sample:
            X = X.sample(sample, random_state=42)
        sv = self.shap_values(X)
        return (
            pd.DataFrame(
                {"feature": self.feature_cols, "mean_abs_shap": np.abs(sv).mean(axis=0)}
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def explain_row(self, row: pd.DataFrame, top_n: int = 6) -> dict:
        """Human-readable drivers for a single forecast."""
        sv = self.shap_values(row)[0]
        base = float(np.ravel(self.explainer.expected_value)[0])
        order = np.argsort(np.abs(sv))[::-1][:top_n]
        drivers = [
            {
                "feature": self.feature_cols[i],
                "value": _coerce(row.iloc[0][self.feature_cols[i]]),
                "shap": round(float(sv[i]), 3),
                "direction": "increases" if sv[i] > 0 else "decreases",
            }
            for i in order
        ]
        return {
            "base_value": round(base, 3),
            "prediction": round(base + float(sv.sum()), 3),
            "top_drivers": drivers,
        }


def _coerce(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if pd.isna(v) else round(float(v), 4)
    return str(v)


def main() -> None:
    """Generate global SHAP artifacts for the trained model."""
    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from src.config import CFG, resolve

    bundle = joblib.load(resolve(CFG.api.model_dir) / "model_bundle.joblib")
    p50 = min(bundle["models"], key=lambda q: abs(q - 0.5))
    model = bundle["models"][p50]
    feature_cols = bundle["feature_cols"]

    df = pd.read_parquet(resolve(CFG.data.processed_path))
    sample = df.sample(min(5000, len(df)), random_state=42)

    ex = ForecastExplainer(model, feature_cols)
    imp = ex.global_importance(sample)
    out = resolve("artifacts")
    imp.to_csv(out / "shap_global_importance.csv", index=False)

    sv = ex.shap_values(sample)
    plt.figure()
    shap.summary_plot(sv, sample[feature_cols], max_display=18, show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, sample[feature_cols], plot_type="bar", max_display=18, show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_importance_bar.png", dpi=130, bbox_inches="tight")
    plt.close()

    print(imp.head(15).to_string(index=False))
    print(f"\nsaved -> {out/'shap_summary.png'}, {out/'shap_global_importance.csv'}")

    print("\nExample single-forecast explanation:")
    import json

    print(json.dumps(ex.explain_row(sample.head(1)), indent=2))


if __name__ == "__main__":
    main()
