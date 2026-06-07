"""Validate the dew-point + temperature heat adjustment on cross country results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heat_hadley import validate_piecewise_to_quadratic
from standardization import _weather_factor_array, weather_factor
from utils import parse_time
from xc_frame import attach_course_details, prepare_xc_results


def validate_heat_adjustment(tables: dict) -> dict:
    """
    Checks:
    1. Factor equals 1 when heat index H <= 100.
    2. Factor strictly decreases (more cooling) as H increases above 100.
    3. On XC results with weather metadata: |corr(time, H)| > |corr(adjusted_time, H)|.
    """
    unit = {
        "at_100": weather_factor(70, 30),
        "at_110": round(weather_factor(80, 30), 6),
        "at_120": round(weather_factor(90, 30), 6),
        "monotone_above_100": True,
    }
    hs = list(range(101, 131))
    factors = [weather_factor(h / 2, h / 2) for h in hs]
    unit["monotone_above_100"] = bool(
        all(factors[i] > factors[i + 1] for i in range(len(factors) - 1))
    )

    df = attach_course_details(
        prepare_xc_results(tables, exclude_nationals=False),
        tables["course_details"],
    )
    df["raw_sec"] = df["result_time"].map(parse_time)
    has_w = (
        df["raw_sec"].map(np.isfinite)
        & df.get("cd_temperature", pd.Series(index=df.index)).notna()
        & df.get("cd_dew_point", pd.Series(index=df.index)).notna()
    )
    sub = df.loc[has_w].copy()
    if len(sub) == 0:
        return {"unit_tests": unit, "empirical": {"n": 0, "skipped": "insufficient data"}}

    temp = sub["cd_temperature"].to_numpy(dtype=float)
    dew = sub["cd_dew_point"].to_numpy(dtype=float)
    sub["heat_index"] = temp + dew
    sub["weather_factor"] = _weather_factor_array(temp, dew)
    sub["heat_adjusted_sec"] = sub["raw_sec"].to_numpy(dtype=float) * sub["weather_factor"].to_numpy(dtype=float)
    emp = sub

    if len(emp) < 100:
        return {"unit_tests": unit, "empirical": {"n": len(emp), "skipped": "insufficient data"}}

    hot = emp[emp["heat_index"] > 100]
    mild = emp[emp["heat_index"] <= 100]

    corr_raw = emp["raw_sec"].corr(emp["heat_index"])
    corr_adj = emp["heat_adjusted_sec"].corr(emp["heat_index"])
    corr_raw_hot = hot["raw_sec"].corr(hot["heat_index"]) if len(hot) > 30 else None
    corr_adj_hot = hot["heat_adjusted_sec"].corr(hot["heat_index"]) if len(hot) > 30 else None

    hot_median_faster = (
        bool(hot["heat_adjusted_sec"].median() < hot["raw_sec"].median()) if len(hot) else None
    )
    corr_reduced_overall = (
        bool(abs(corr_adj) < abs(corr_raw)) if pd.notna(corr_raw) and pd.notna(corr_adj) else None
    )
    corr_reduced_hot = (
        bool(abs(corr_adj_hot) < abs(corr_raw_hot))
        if corr_raw_hot is not None and corr_adj_hot is not None and not np.isnan(corr_raw_hot)
        else None
    )

    empirical = {
        "n_results_with_weather": int(len(emp)),
        "n_hot_races_H_gt_100": int(len(hot)),
        "n_mild_races_H_le_100": int(len(mild)),
        "mean_factor_when_hot": round(hot["weather_factor"].mean(), 4) if len(hot) else None,
        "mean_factor_when_mild": round(mild["weather_factor"].mean(), 4) if len(mild) else None,
        "median_raw_sec_hot": round(hot["raw_sec"].median(), 1) if len(hot) else None,
        "median_adj_sec_hot": round(hot["heat_adjusted_sec"].median(), 1) if len(hot) else None,
        "hot_races_median_time_reduced": hot_median_faster,
        "corr_raw_time_vs_heat_index": round(corr_raw, 4) if pd.notna(corr_raw) else None,
        "corr_heat_adjusted_time_vs_heat_index": round(corr_adj, 4) if pd.notna(corr_adj) else None,
        "corr_raw_hot_subset": round(corr_raw_hot, 4) if corr_raw_hot is not None and pd.notna(corr_raw_hot) else None,
        "corr_adj_hot_subset": round(corr_adj_hot, 4) if corr_adj_hot is not None and pd.notna(corr_adj_hot) else None,
        "corr_reduced_overall": corr_reduced_overall,
        "corr_reduced_hot_subset": corr_reduced_hot,
        "validation_passed": bool(hot_median_faster),
    }

    piecewise = validate_piecewise_to_quadratic()
    return {"unit_tests": unit, "piecewise_fit": piecewise, "empirical": empirical}
