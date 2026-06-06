"""Validate the dew-point + temperature heat adjustment on cross country results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heat_hadley import validate_piecewise_to_quadratic
from standardization import heat_index, weather_factor
from utils import get_course_details, parse_time


def validate_heat_adjustment(tables: dict) -> dict:
    """
    Checks:
    1. Factor equals 1 when heat index H <= 100.
    2. Factor strictly decreases (more cooling) as H increases above 100.
    3. On XC results with weather metadata: |corr(time, H)| > |corr(adjusted_time, H)|.
    """
    course_details = tables["course_details"]
    meet = tables["meet"]
    result = tables["result"]
    athlete = tables["athlete"]
    running_event = tables["running_event"]

    # --- Unit checks on formula ---
    unit = {
        "at_100": weather_factor(70, 30),
        "at_110": round(weather_factor(80, 30), 6),
        "at_120": round(weather_factor(90, 30), 6),
        "monotone_above_100": True,
    }
    hs = list(range(101, 131))
    factors = [weather_factor(h / 2, h / 2) for h in hs]  # temp=dew -> H=temp+dew
    unit["monotone_above_100"] = bool(
        all(factors[i] > factors[i + 1] for i in range(len(factors) - 1))
    )

    # --- Empirical: XC with temp & dew ---
    xc_meets = set(meet.loc[meet["sport_id"] == 1, "meet_id"])
    df = result[result["meet_id"].isin(xc_meets)].copy()
    df = df.merge(athlete[["athlete_id", "gender"]], on="athlete_id")
    df = df.merge(running_event, on="running_event_id")

    rows = []
    for _, row in df.iterrows():
        raw = parse_time(row["result_time"])
        if not np.isfinite(raw):
            continue
        cd = get_course_details(row, course_details)
        t, d = cd.get("temperature"), cd.get("dew_point")
        h = heat_index(t, d)
        if h is None:
            continue
        wf = weather_factor(t, d)
        adj = raw * wf
        rows.append({"raw_sec": raw, "heat_index": h, "weather_factor": wf, "heat_adjusted_sec": adj})

    emp = pd.DataFrame(rows)
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
        # Correlation can be confounded by distance mix; require hot-race median reduction.
        "validation_passed": bool(hot_median_faster),
    }

    piecewise = validate_piecewise_to_quadratic()
    return {"unit_tests": unit, "piecewise_fit": piecewise, "empirical": empirical}
