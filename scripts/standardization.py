"""Performance standardization factors and XC pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from altitude import apply_meet_altitude
from config import (
    HEAT_QUADRATIC_COEFF,
    MIN_ELEVATION_GRADE_DISTANCE_M,
    RIEGEL_B_MEN,
    RIEGEL_B_WOMEN,
    XC_TARGET_F,
    XC_TARGET_M,
)
from events import parse_event_distance_m
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
    """Maurer multipliers from gain/loss *grades in percent* (not feet)."""
    g = 0.0 if gain_pct is None or (isinstance(gain_pct, float) and np.isnan(gain_pct)) else float(gain_pct)
    l = 0.0 if loss_pct is None or (isinstance(loss_pct, float) and np.isnan(loss_pct)) else float(loss_pct)
    return (1.04**g) * (0.9633**l)


def _distance_for_elevation_grade_m(d_actual, d_reported) -> float | None:
    """Pick a plausible course length (m) for ft→grade% conversion."""
    candidates = []
    for d in (d_actual, d_reported):
        if d is None or (isinstance(d, float) and np.isnan(d)):
            continue
        try:
            v = float(d)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v >= MIN_ELEVATION_GRADE_DISTANCE_M:
            candidates.append(v)
    if not candidates:
        return None
    return max(candidates)


def _elevation_feet_to_grade_pct(elevation_ft, distance_m: float | None) -> float | None:
    """Convert absolute elevation change (ft) to average grade percent."""
    if elevation_ft is None or (isinstance(elevation_ft, float) and np.isnan(elevation_ft)):
        return None
    if distance_m is None or not np.isfinite(distance_m) or distance_m <= 0:
        return None
    try:
        ft = float(elevation_ft)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ft) or ft < 0:
        return None
    distance_ft = distance_m * 3.280839895
    if distance_ft <= 0:
        return None
    return 100.0 * ft / distance_ft


def elevation_factor_from_metadata(
    gain_ft,
    loss_ft,
    *,
    distance_m: float | None,
    d_reported: float | None = None,
) -> float:
    """Apply Maurer factors using stored gain/loss in feet and course length in meters."""
    dist = _distance_for_elevation_grade_m(distance_m, d_reported)
    if dist is None:
        return 1.0
    g = _elevation_feet_to_grade_pct(gain_ft, dist)
    l = _elevation_feet_to_grade_pct(loss_ft, dist)
    if g is None and l is None:
        return 1.0
    return elevation_factor(g or 0.0, l or 0.0)


def _course_details_for_row(row, course_details_df) -> dict:
    """Prefer merged ``cd_*`` columns (from :func:`xc_frame.attach_course_details`)."""
    if hasattr(row, "index") and any(str(c).startswith("cd_") for c in row.index):
        rec = {
            col[3:]: row[col]
            for col in row.index
            if str(col).startswith("cd_") and pd.notna(row[col])
        }
        if rec:
            return rec
    return get_course_details(row, course_details_df)


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

    cd = _course_details_for_row(row, course_details_df)
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
        t *= elevation_factor_from_metadata(
            gain, loss, distance_m=d_actual, d_reported=d_reported
        )
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
    cd = _course_details_for_row(row, course_details_df)
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
        t *= elevation_factor_from_metadata(
            gain, loss, distance_m=d_actual, d_reported=d_reported
        )
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
    cd = _course_details_for_row(row, course_details_df)
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


def _weather_factor_array(
    temp: np.ndarray, dew: np.ndarray, k: float | None = None
) -> np.ndarray:
    coeff = HEAT_QUADRATIC_COEFF if k is None else k
    heat = temp + dew
    pct = coeff * np.square(heat - 100.0)
    factors = 1.0 - pct / 100.0
    factors = np.where(heat <= 100.0, 1.0, factors)
    invalid = ~np.isfinite(temp) | ~np.isfinite(dew)
    return np.where(invalid, 1.0, factors)


def _elevation_factor_array(
    gain_ft: np.ndarray,
    loss_ft: np.ndarray,
    distance_m: np.ndarray,
    d_reported: np.ndarray,
) -> np.ndarray:
    n = len(gain_ft)
    out = np.ones(n, dtype=float)
    for i in range(n):
        dm = distance_m[i] if np.isfinite(distance_m[i]) else np.nan
        dr = d_reported[i] if np.isfinite(d_reported[i]) else np.nan
        out[i] = elevation_factor_from_metadata(
            gain_ft[i] if np.isfinite(gain_ft[i]) else None,
            loss_ft[i] if np.isfinite(loss_ft[i]) else None,
            distance_m=dm,
            d_reported=dr,
        )
    return out


def _riegel_convert_array(
    seconds: np.ndarray,
    d_actual: np.ndarray,
    d_target: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    ratio = np.divide(
        d_target,
        d_actual,
        out=np.full_like(seconds, np.nan, dtype=float),
        where=(d_actual > 0) & (d_target > 0),
    )
    out = seconds * np.power(ratio, b)
    bad = (
        ~np.isfinite(seconds)
        | ~np.isfinite(d_actual)
        | ~np.isfinite(d_target)
        | (d_actual <= 0)
        | (d_target <= 0)
    )
    out[bad] = np.nan
    return out


def _course_record_from_row(row: pd.Series) -> dict:
    return {
        col[3:]: row[col]
        for col in row.index
        if col.startswith("cd_") and pd.notna(row[col])
    }


def _meet_elev_array(df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        row = df.iloc[i]
        z = meet_elevation_ft_from_row(row, _course_record_from_row(row))
        if z is not None:
            out[i] = z
    return out


def _baro_hpa_array(df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        pb = barometric_pressure_hpa_from_record(_course_record_from_row(df.iloc[i]))
        if pb is not None:
            out[i] = pb
    return out


def _apply_altitude_array(
    times: np.ndarray,
    event_names: pd.Series,
    elevations: np.ndarray,
    genders: pd.Series,
    pb_hpa: np.ndarray,
) -> np.ndarray:
    out = times.copy()
    for i in range(len(times)):
        if not np.isfinite(times[i]):
            continue
        elev = elevations[i]
        if not np.isfinite(elev) or elev <= 0:
            continue
        pb = pb_hpa[i] if np.isfinite(pb_hpa[i]) else None
        out[i] = apply_meet_altitude(
            float(times[i]),
            event_names.iat[i],
            float(elev),
            genders.iat[i],
            barometric_pressure_hpa=pb,
        )
    return out


def compute_xc_times(
    df: pd.DataFrame,
    *,
    heat_k: float | None = None,
    riegel_b_men: float | None = None,
    riegel_b_women: float | None = None,
    riegel_b_unified: float | None = None,
) -> pd.DataFrame:
    """Vectorized XC clocks on a frame from :func:`xc_frame.build_xc_frame`.

    Adds ``raw_sec``, ``converted_sec``, and ``standardized_sec`` columns.
    Optional Riegel overrides support sensitivity analysis (``riegel_b_unified``
    applies one exponent to all genders).
    """
    out = df.copy()
    raw = out["result_time"].map(parse_time).to_numpy(dtype=float)
    out["raw_sec"] = raw

    is_men = out["gender"].eq("M").to_numpy()
    b_m = riegel_b_men if riegel_b_men is not None else RIEGEL_B_MEN
    b_w = riegel_b_women if riegel_b_women is not None else RIEGEL_B_WOMEN
    if riegel_b_unified is not None:
        b_m = b_w = riegel_b_unified
    b = np.where(is_men, b_m, b_w)
    target = np.where(is_men, XC_TARGET_M, XC_TARGET_F)
    d_reported = out["event_name"].map(parse_event_distance_m).to_numpy(dtype=float)

    if "cd_estimated_course_distance" in out.columns:
        d_actual = out["cd_estimated_course_distance"].to_numpy(dtype=float)
    else:
        d_actual = np.full(len(out), np.nan)
    d_actual = np.where(np.isfinite(d_actual), d_actual, d_reported)

    valid_dist = np.isfinite(d_reported) & np.isfinite(d_actual)
    length_fix = valid_dist & (d_reported > 0) & (d_actual > 0) & (np.abs(d_actual - d_reported) > 1)

    converted = raw.copy()
    if length_fix.any():
        converted[length_fix] = _riegel_convert_array(
            raw[length_fix], d_actual[length_fix], d_reported[length_fix], b[length_fix]
        )
    converted = np.where(
        valid_dist,
        _riegel_convert_array(converted, d_actual, target, b),
        raw,
    )
    out["converted_sec"] = converted

    temp = (
        out["cd_temperature"].to_numpy(dtype=float)
        if "cd_temperature" in out.columns
        else np.full(len(out), np.nan)
    )
    dew = (
        out["cd_dew_point"].to_numpy(dtype=float)
        if "cd_dew_point" in out.columns
        else np.full(len(out), np.nan)
    )
    gain = (
        out["cd_elevation_gain"].to_numpy(dtype=float)
        if "cd_elevation_gain" in out.columns
        else np.full(len(out), np.nan)
    )
    loss = (
        out["cd_elevation_loss"].to_numpy(dtype=float)
        if "cd_elevation_loss" in out.columns
        else np.full(len(out), np.nan)
    )
    pb = _baro_hpa_array(out)
    elev = _meet_elev_array(out)

    std = raw.copy()
    std = std * _elevation_factor_array(gain, loss, d_actual, d_reported)
    std = _apply_altitude_array(std, out["event_name"], elev, out["gender"], pb)
    if length_fix.any():
        std[length_fix] = _riegel_convert_array(
            std[length_fix], d_actual[length_fix], d_reported[length_fix], b[length_fix]
        )
    std = _riegel_convert_array(std, d_actual, target, b)
    std = std * _weather_factor_array(temp, dew, k=heat_k)
    std = np.where(valid_dist, std, raw)
    out["standardized_sec"] = std
    return out
