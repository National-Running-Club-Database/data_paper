"""Sensitivity of XC validation metrics to heat quadratic coefficient k."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import HEAT_QUADRATIC_COEFF
from progress import iter_progress
from standardization import compute_xc_times
from validation_improvement import _median_improvement_by_gender
from xc_frame import build_xc_frame


DEFAULT_K_VALUES = (0.001, 0.0016, 0.002)


def _variance_by_gender(perf: pd.DataFrame, min_meets: int) -> dict[str, dict]:
    out = {}
    for gender, label in [("F", "women"), ("M", "men")]:
        g = perf[perf["gender"] == gender]
        athlete_season_stats = []
        for (_, _), grp in g.groupby(["athlete_id", "season"]):
            if grp["meet_id"].nunique() < min_meets:
                continue
            athlete_season_stats.append(
                {"raw_sd": grp["raw"].std(), "std_sd": grp["standardized"].std()}
            )
        if not athlete_season_stats:
            out[label] = {}
            continue
        stats_df = pd.DataFrame(athlete_season_stats)
        med_raw = float(stats_df["raw_sd"].median())
        med_std = float(stats_df["std_sd"].median())
        out[label] = {
            "n_athlete_seasons": len(stats_df),
            "median_raw_sd_sec": round(med_raw, 2),
            "median_std_sd_sec": round(med_std, 2),
            "pct_reduction_std_vs_raw": round(100 * (1 - med_std / med_raw), 1),
        }
    return out


def _performance_frame(timed: pd.DataFrame) -> pd.DataFrame:
    perf = timed[np.isfinite(timed["raw_sec"]) & np.isfinite(timed["standardized_sec"])].copy()
    perf = perf.rename(columns={"raw_sec": "raw", "standardized_sec": "standardized"})
    return perf.dropna(subset=["season"])


def _standardized_improvement_frame(timed: pd.DataFrame) -> pd.DataFrame:
    out = timed.loc[np.isfinite(timed["standardized_sec"])].copy()
    out = out.rename(columns={"standardized_sec": "time"})
    return out.dropna(subset=["race_date"])


def k_sensitivity_summary(
    tables: dict,
    k_values: tuple[float, ...] | None = None,
    min_meets: int = 3,
) -> dict:
    """Variance reduction and improvement inflation vs. heat k (converted-only unchanged)."""
    k_values = k_values or DEFAULT_K_VALUES
    base = build_xc_frame(tables, exclude_nationals=True)

    timed_conv = compute_xc_times(base, heat_k=HEAT_QUADRATIC_COEFF)
    conv = timed_conv.loc[np.isfinite(timed_conv["converted_sec"])].copy()
    conv = conv.rename(columns={"converted_sec": "time"})
    conv_by_gender = _median_improvement_by_gender(conv.dropna(subset=["race_date"]))

    by_k = []
    for k in iter_progress(k_values, desc="k sensitivity", unit="k"):
        timed = compute_xc_times(base, heat_k=k)
        perf = _performance_frame(timed)
        std = _standardized_improvement_frame(timed)
        std_by_gender = _median_improvement_by_gender(std)
        inflation = {}
        for label in ("men", "women"):
            c = conv_by_gender.get(label, {}).get("median_improvement_sec")
            s = std_by_gender.get(label, {}).get("median_improvement_sec")
            if c is not None and s is not None and s > 0:
                inflation[label] = {
                    "median_conv_sec": c,
                    "median_std_sec": s,
                    "pct_inflation_vs_full": round(100 * (c / s - 1), 0),
                }
            else:
                inflation[label] = {}
        by_k.append(
            {
                "k": k,
                "is_deployed": abs(k - HEAT_QUADRATIC_COEFF) < 1e-12,
                "variance_by_gender": _variance_by_gender(perf, min_meets),
                "improvement_inflation": inflation,
            }
        )

    return {"k_values": list(k_values), "by_k": by_k}
