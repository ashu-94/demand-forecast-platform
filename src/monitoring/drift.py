"""Drift detection and retraining triggers.

Three distinct failure modes get monitored, because they fail at different
times and mean different things:

1. **Feature drift**  -- inputs moved (new store, price reset, promo strategy
   change). PSI + KS. Early warning; the model may still be fine.
2. **Prediction drift** -- outputs moved without ground truth arriving yet.
   In forecasting you wait days-to-weeks for actuals, so this is often the
   only signal you have in time to act.
3. **Performance drift** -- WAPE on realised actuals degraded vs the training
   baseline. Slowest but definitive. This is what actually triggers a retrain.

PSI is implemented directly rather than pulled from a library so the bin edges
are frozen from the reference window -- recomputing edges on current data is a
common bug that hides drift.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.config import CFG
from src.models.metrics import bias, wape

log = logging.getLogger(__name__)

PSI_BANDS = [(0.10, "none"), (0.20, "moderate"), (float("inf"), "severe")]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with bin edges frozen from the reference."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 20 or len(cur) < 20:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    r = np.histogram(ref, bins=edges)[0].astype(float)
    c = np.histogram(cur, bins=edges)[0].astype(float)
    r, c = r / max(r.sum(), 1), c / max(c.sum(), 1)
    eps = 1e-6
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    r = reference.value_counts(normalize=True)
    c = current.value_counts(normalize=True)
    cats = r.index.union(c.index)
    eps = 1e-6
    rv = np.clip(r.reindex(cats).fillna(0).to_numpy(), eps, None)
    cv = np.clip(c.reindex(cats).fillna(0).to_numpy(), eps, None)
    return float(np.sum((cv - rv) * np.log(cv / rv)))


def psi_band(value: float) -> str:
    for threshold, label in PSI_BANDS:
        if value < threshold:
            return label
    return "severe"


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    feature_drift: pd.DataFrame = field(default_factory=pd.DataFrame)
    prediction_drift: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    triggers: list[str] = field(default_factory=list)

    @property
    def retrain_required(self) -> bool:
        return len(self.triggers) > 0

    def summary(self) -> str:
        lines = [
            f"reference={self.n_reference:,}  current={self.n_current:,}",
        ]
        if not self.feature_drift.empty:
            drifted = self.feature_drift[self.feature_drift["band"] != "none"]
            lines.append(f"drifted features: {len(drifted)}/{len(self.feature_drift)}")
        if self.prediction_drift:
            lines.append(
                f"prediction PSI={self.prediction_drift.get('psi', 0):.4f} "
                f"({self.prediction_drift.get('band')})"
            )
        if self.performance:
            lines.append(
                f"WAPE baseline={self.performance.get('baseline_wape', float('nan')):.4f} "
                f"current={self.performance.get('current_wape', float('nan')):.4f} "
                f"({self.performance.get('relative_change', 0) * 100:+.1f}%)"
            )
        lines.append(
            "RETRAIN REQUIRED: " + ("yes -> " + "; ".join(self.triggers) if self.triggers else "no")
        )
        return "\n".join(lines)


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in reference.columns or f not in current.columns:
            continue
        ref_s, cur_s = reference[f], current[f]
        is_cat = str(ref_s.dtype) in ("object", "category", "string") or ref_s.dtype == bool

        if is_cat:
            value = categorical_psi(ref_s.astype(str), cur_s.astype(str))
            try:
                cats = sorted(set(ref_s.astype(str)) | set(cur_s.astype(str)))
                table = np.array(
                    [
                        ref_s.astype(str).value_counts().reindex(cats).fillna(0).to_numpy(),
                        cur_s.astype(str).value_counts().reindex(cats).fillna(0).to_numpy(),
                    ]
                )
                p = float(stats.chi2_contingency(table + 1)[1])
            except Exception:  # pragma: no cover
                p = float("nan")
            test = "chi2"
        else:
            value = psi(ref_s.to_numpy(), cur_s.to_numpy())
            r = ref_s.dropna().to_numpy()
            c = cur_s.dropna().to_numpy()
            p = float(stats.ks_2samp(r, c).pvalue) if len(r) > 20 and len(c) > 20 else float("nan")
            test = "ks"

        rows.append(
            {
                "feature": f,
                "type": "categorical" if is_cat else "numeric",
                "psi": round(value, 5),
                "band": psi_band(value),
                "test": test,
                "p_value": None if p != p else round(p, 6),
                "significant": bool(p == p and p < alpha),
                "ref_mean": None if is_cat else round(float(ref_s.mean()), 4),
                "cur_mean": None if is_cat else round(float(cur_s.mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def run_drift_check(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    pred_col: str | None = None,
    target_col: str | None = None,
    baseline_wape: float | None = None,
    cfg=CFG,
    wape_tolerance: float = 0.15,
) -> DriftReport:
    mon = cfg.monitoring
    rep = DriftReport(n_reference=len(reference), n_current=len(current))

    rep.feature_drift = detect_feature_drift(
        reference, current, features, alpha=float(mon.ks_alpha)
    )

    if pred_col and pred_col in reference.columns and pred_col in current.columns:
        v = psi(reference[pred_col].to_numpy(), current[pred_col].to_numpy())
        rep.prediction_drift = {
            "psi": round(v, 5),
            "band": psi_band(v),
            "ref_mean": round(float(reference[pred_col].mean()), 3),
            "cur_mean": round(float(current[pred_col].mean()), 3),
        }

    if pred_col and target_col and target_col in current.columns:
        cur_wape = wape(current[target_col], current[pred_col])
        rep.performance = {
            "current_wape": round(cur_wape, 5),
            "current_bias": round(bias(current[target_col], current[pred_col]), 5),
        }
        if baseline_wape:
            rel = (cur_wape - baseline_wape) / baseline_wape
            rep.performance["baseline_wape"] = round(baseline_wape, 5)
            rep.performance["relative_change"] = round(rel, 5)
            if rel > wape_tolerance:
                rep.triggers.append(
                    f"WAPE degraded {rel:.1%} vs baseline (tolerance {wape_tolerance:.0%})"
                )

    severe = rep.feature_drift[rep.feature_drift["psi"] >= float(mon.psi_threshold)]
    if len(severe) >= 3:
        rep.triggers.append(
            f"{len(severe)} features above PSI {mon.psi_threshold}: "
            + ", ".join(severe["feature"].head(5))
        )
    if rep.prediction_drift.get("psi", 0) >= float(mon.psi_threshold):
        rep.triggers.append(f"prediction PSI {rep.prediction_drift['psi']:.3f}")

    return rep
