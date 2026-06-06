"""Performance standardization factors and XC pipeline."""

from __future__ import annotations

import numpy as np

from config import HEAT_QUADRATIC_COEFF, RIEGEL_B_MEN, RIEGEL_B_WOMEN, XC_TARGET_F, XC_TARGET_M
from utils import (
    apply_altitude_to_seconds,
    barometric_pressure_hpa_from_record,
    get_course_details,
    get_event_dist,
    meet_elevation_ft_from_row,
    parse_time,
)


def riegel_convert(seconds: float, d_actual: float, d_target: float, b: float) -> float:
    if not np.isfinite(seconds) or d_actual <= 0 or d_target <= 0:
        return np.nan
    return seconds * (d_target / d_actual) ** b


def weather_factor(temp, dew, k: float | None = None) -> float:
    """Multiplicative factor; <1 when heat index (temp+dew °F) exceeds 100."""
    if temp is None or dew is None or (isinstance(temp, float) and np.isnan(temp)):
        return 1.0
    if isinstance(dew, float) and np.isnan(dew):
        return 1.0
    heat = float(temp) + float(dew)
    if heat <= 100:
        return 1.0
    coeff = HEAT_QUADRATIC_COEFF if k is None else k
    pct = coeff * (heat - 100) ** 2
    return 1.0 - pct / 100.0


def heat_index(temp, dew) -> float | None:
    if temp is None or dew is None:
        return None
    try:
        t, d = float(temp), float(dew)
    except (TypeError, ValueError):
        return None
    if np.isnan(t) or np.isnan(d):
        return None
    return t + d


def elevation_factor(gain_pct, loss_pct) -> float:
    g = 0.0 if gain_pct is None or (isinstance(gain_pct, float) and np.isnan(gain_pct)) else float(gain_pct)
    l = 0.0 if loss_pct is None or (isinstance(loss_pct, float) and np.isnan(loss_pct)) else float(loss_pct)
    return (1.04**g) * (0.9633**l)


def riegel_exponent(gender: str) -> float:
    return RIEGEL_B_MEN if gender == "M" else RIEGEL_B_WOMEN


def xc_target_distance(gender: str) -> float:
    return XC_TARGET_M if gender == "M" else XC_TARGET_F


def standardize_xc_row(
    row,
    course_details_df,
    b: float | None = None,
    heat_k: float | None = None,
) -> tuple[float, float]:
    """Return (raw_seconds, standardized_seconds) for a cross country result."""
    if b is None:
        b = riegel_exponent(row["gender"])
    raw = parse_time(row["result_time"])
    if not np.isfinite(raw):
        return np.nan, np.nan

    cd = get_course_details(row, course_details_df)
    d_reported = get_event_dist(row.get("event_name"))
    d_actual = cd.get("estimated_course_distance")
    if d_actual is None or (isinstance(d_actual, float) and np.isnan(d_actual)):
        d_actual = d_reported
    if d_reported is None or d_actual is None:
        return raw, raw

    target = xc_target_distance(row["gender"])
    t = raw
    t *= weather_factor(cd.get("temperature"), cd.get("dew_point"), k=heat_k)
    gain, loss = cd.get("elevation_gain"), cd.get("elevation_loss")
    if (gain is not None and not (isinstance(gain, float) and np.isnan(gain))) or (
        loss is not None and not (isinstance(loss, float) and np.isnan(loss))
    ):
        t *= elevation_factor(gain, loss)
    meet_elev = meet_elevation_ft_from_row(row, cd)
    pb_hpa = barometric_pressure_hpa_from_record(cd)
    t = apply_altitude_to_seconds(
        t,
        row.get("event_name"),
        meet_elev,
        row.get("gender"),
        barometric_pressure_hpa=pb_hpa,
    )
    if d_reported > 0 and d_actual > 0 and abs(d_actual - d_reported) > 1:
        t = riegel_convert(t, d_actual, d_reported, b)
    std = riegel_convert(t, d_actual, target, b)
    return raw, std


def pre_weather_xc(row, course_details_df, b: float | None = None) -> float:
    """Distance + elevation + course-length adjusted time; no weather factor."""
    if b is None:
        b = riegel_exponent(row["gender"])
    raw = parse_time(row["result_time"])
    if not np.isfinite(raw):
        return np.nan
    cd = get_course_details(row, course_details_df)
    d_reported = get_event_dist(row.get("event_name"))
    d_actual = cd.get("estimated_course_distance")
    if d_actual is None or (isinstance(d_actual, float) and np.isnan(d_actual)):
        d_actual = d_reported
    if d_reported is None or d_actual is None:
        return np.nan
    target = xc_target_distance(row["gender"])
    t = raw
    gain, loss = cd.get("elevation_gain"), cd.get("elevation_loss")
    if (gain is not None and not (isinstance(gain, float) and np.isnan(gain))) or (
        loss is not None and not (isinstance(loss, float) and np.isnan(loss))
    ):
        t *= elevation_factor(gain, loss)
    if d_reported > 0 and d_actual > 0 and abs(d_actual - d_reported) > 1:
        t = riegel_convert(t, d_actual, d_reported, b)
    return riegel_convert(t, d_actual, target, b)


def apply_heat_factor(seconds: float, h: float | None, k: float) -> float:
    if not np.isfinite(seconds) or h is None or h <= 100:
        return seconds
    pct = k * (h - 100) ** 2
    return seconds * (1.0 - pct / 100.0)


def converted_only_xc(row, course_details_df, b: float | None = None) -> float:
    if b is None:
        b = riegel_exponent(row["gender"])
    raw = parse_time(row["result_time"])
    if not np.isfinite(raw):
        return np.nan
    cd = get_course_details(row, course_details_df)
    d_reported = get_event_dist(row.get("event_name"))
    d_actual = cd.get("estimated_course_distance")
    if d_actual is None or (isinstance(d_actual, float) and np.isnan(d_actual)):
        d_actual = d_reported
    if d_reported is None or d_actual is None:
        return raw
    target = xc_target_distance(row["gender"])
    t = raw
    if d_reported > 0 and d_actual > 0 and abs(d_actual - d_reported) > 1:
        t = riegel_convert(t, d_actual, d_reported, b)
    return riegel_convert(t, d_actual, target, b)
