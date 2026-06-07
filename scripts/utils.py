from __future__ import annotations

import math
import re

import pandas as pd

from altitude import (
    altitude_power_percent_sea_level,
    apply_meet_altitude,
    barometric_pressure_hpa_from_record,
    sea_level_time_seconds,
)
from config import HEAT_QUADRATIC_COEFF, TRACK_OUTDOOR_REFERENCE_LAP_M, WIND_MAX_APPLY_MPS
from events import applies_altitude_conversion, event_category, parse_event_distance_m
from schema import meet_altitude_ft_from_record
from track_index import ncaa_index_multiplier, oversized_to_flat_multiplier
from wind_index import wind_delta_seconds_at_speed


def parse_time(time_str):
    if pd.isna(time_str):
        return float("nan")
    parts = str(time_str).split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        else:
            return float(time_str)
    except Exception:
        return float("nan")


def format_parsed_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:05.2f}"
    elif minutes > 0:
        return f"{minutes}:{seconds:05.2f}"
    else:
        return f"{seconds:05.2f}"


def _finite_elevation_feet(elevation_feet) -> float | None:
    if elevation_feet is None:
        return None
    try:
        z = float(elevation_feet)
    except (TypeError, ValueError):
        return None
    if pd.isna(z) or z <= 0:
        return None
    return z


def altitude_power_percent(elevation_ft) -> float:
    """Max aerobic power at venue altitude as % of sea level (Peronnet 1991 Eq. 3)."""
    z = _finite_elevation_feet(elevation_ft)
    if z is None:
        return 100.0
    return altitude_power_percent_sea_level(z)


def altitude_factor(elevation_ft, event_name: str | None = None) -> float:
    """MAP ratio for event-level checks; unity when elevation missing or not a running event."""
    if not applies_altitude_conversion(event_name):
        return 1.0
    z = _finite_elevation_feet(elevation_ft)
    if z is None:
        return 1.0
    return altitude_power_percent_sea_level(z) / 100.0


def altitude_sea_level_time_seconds(
    time_sec: float,
    distance_m: float,
    elevation_ft,
    gender: str = "M",
    *,
    barometric_pressure_hpa: float | None = None,
) -> float:
    """Sea-level equivalent time: t_raw * f_alt (Peronnet; optional race-time pressure hPa)."""
    return sea_level_time_seconds(
        time_sec,
        distance_m,
        elevation_ft,
        gender,
        barometric_pressure_hpa=barometric_pressure_hpa,
    )


def meet_elevation_ft_from_row(row, course_details=None):
    """Meet venue altitude (ft) from merged row or ``course_details``."""
    return meet_altitude_ft_from_record(row, course_details)


def apply_altitude_to_seconds(
    time_sec: float,
    event_name: str | None,
    elevation_ft,
    gender: str = "M",
    *,
    barometric_pressure_hpa: float | None = None,
    warn_on_orphan_pressure: bool = True,
) -> float:
    """Sea-level time via Peronnet when event qualifies and **meet elevation** is set.

    Elevation only → full $f_{\\mathrm{alt}}$ with $\\rho$ from $P_b(z)$.
    Elevation + OpenWeather ``barometric_pressure`` (hPa) → MAP from elevation, $\\rho$ from race pressure.
    Pressure without elevation → unchanged time (pressure ignored).
    """
    return apply_meet_altitude(
        time_sec,
        event_name,
        elevation_ft,
        gender,
        barometric_pressure_hpa=barometric_pressure_hpa,
        warn_on_orphan_pressure=warn_on_orphan_pressure,
    )


def _normalize_lap_length_m(lap_length_m) -> float | None:
    if lap_length_m is None or (isinstance(lap_length_m, float) and pd.isna(lap_length_m)):
        return None
    try:
        lap = float(lap_length_m)
    except (TypeError, ValueError):
        return None
    if lap <= 0:
        return None
    return lap


def _parse_banked(banked) -> bool:
    if banked is None or (isinstance(banked, float) and pd.isna(banked)):
        return False
    if isinstance(banked, bool):
        return banked
    s = str(banked).strip().lower()
    return s in ("1", "true", "yes", "y", "banked", "bt")


def _is_indoor_sport(sport_name: str | None) -> bool:
    if sport_name is None:
        return False
    return "indoor" in str(sport_name).lower()


def _is_outdoor_track(sport_name: str | None) -> bool:
    if sport_name is None:
        return False
    s = str(sport_name).lower()
    return "outdoor" in s and "track" in s


def parse_wind_mps(wind) -> float | None:
    """Parse result.wind (m/s); positive = tailwind (aiding) per TFRRS convention."""
    if wind is None or (isinstance(wind, float) and pd.isna(wind)):
        return None
    s = str(wind).strip().replace("+", "")
    if not s or s.lower() in ("n/a", "na", "none"):
        return None
    try:
        w = float(s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(w):
        return None
    return w


def applies_wind_conversion(event_name: str | None, sport_name: str | None) -> bool:
    """Outdoor track sprints and hurdles (<=400 m) when wind is recorded."""
    if not _is_outdoor_track(sport_name):
        return False
    if event_name is None or (isinstance(event_name, float) and pd.isna(event_name)):
        return False
    return event_category(str(event_name)) == "sprint"


def wind_calm_equivalent_seconds(
    time_sec: float,
    wind_mps: float,
    event_distance_m: float,
    gender: str,
) -> float:
    """Calm-equivalent time: $t_{\\mathrm{calm}} = t + \\Delta t$ with event-specific $\\Delta t$ (wind_index)."""
    if (
        time_sec is None
        or not math.isfinite(float(time_sec))
        or time_sec <= 0
        or event_distance_m is None
        or event_distance_m <= 0
    ):
        return time_sec

    w = float(wind_mps)
    if abs(w) > WIND_MAX_APPLY_MPS:
        w = math.copysign(WIND_MAX_APPLY_MPS, w)

    delta = wind_delta_seconds_at_speed(w, float(event_distance_m), gender)
    return float(time_sec) + delta


def wind_factor(
    event_name: str | None,
    gender: str,
    wind_mps,
    sport_name: str | None = None,
) -> float:
    """Multiplicative $f_{\\mathrm{wind}} = t_{\\mathrm{calm}}/t_{\\mathrm{raw}}$; unity when not applicable."""
    if not applies_wind_conversion(event_name, sport_name):
        return 1.0
    w = parse_wind_mps(wind_mps)
    if w is None:
        return 1.0
    d = get_event_dist(event_name)
    if d is None:
        return 1.0
    t_calm = wind_calm_equivalent_seconds(1.0, w, d, gender)
    return t_calm


def apply_wind_to_seconds(
    time_sec: float,
    event_name: str | None,
    gender: str,
    wind_mps,
    sport_name: str | None = None,
) -> float:
    """Apply $f_{\\mathrm{wind}}$ for outdoor sprint/hurdle results."""
    if time_sec is None or not math.isfinite(float(time_sec)) or time_sec <= 0:
        return time_sec
    if not applies_wind_conversion(event_name, sport_name):
        return time_sec
    w = parse_wind_mps(wind_mps)
    if w is None:
        return time_sec
    d = get_event_dist(event_name)
    if d is None:
        return time_sec
    return wind_calm_equivalent_seconds(time_sec, w, d, gender)


def track_length_factor_to_outdoor_flat(
    event_distance_m: float,
    gender: str,
    *,
    lap_length_m,
    indoor: bool,
) -> float:
    """$f_{\\mathrm{len}}$: NCAA tabulated $\\alpha(D)$, nearest standard event distance."""
    if event_distance_m is None or event_distance_m <= 0:
        return 1.0
    lap = _normalize_lap_length_m(lap_length_m)
    if lap is None:
        return 1.0
    if not indoor and lap == TRACK_OUTDOOR_REFERENCE_LAP_M:
        return 1.0

    d = float(event_distance_m)
    if lap < 200.0:
        c_us = ncaa_index_multiplier(d, gender, "undersized_to_flat")
        return 1.0 / c_us
    return oversized_to_flat_multiplier(d, gender)


def track_banking_factor_to_outdoor_flat(
    event_distance_m: float,
    gender: str,
    *,
    banked,
    indoor: bool,
) -> float:
    """$f_{\\mathrm{bank}}$: $1/\\alpha_{\\mathrm{fb}}(D)$ when banked indoor (NCAA charts)."""
    if event_distance_m is None or event_distance_m <= 0:
        return 1.0
    if not _parse_banked(banked):
        return 1.0
    if not indoor:
        return 1.0

    c_fb = ncaa_index_multiplier(float(event_distance_m), gender, "flat_to_banked")
    return 1.0 / c_fb


def track_venue_factor_to_outdoor_flat(
    event_name: str | None,
    gender: str,
    *,
    lap_length_m=None,
    banked=None,
    sport_name: str | None = None,
) -> float:
    """Combined $f_{\\mathrm{track}} = f_{\\mathrm{len}} \\cdot f_{\\mathrm{bank}}$ (unity if N/A)."""
    if event_name is None or (isinstance(event_name, float) and pd.isna(event_name)):
        return 1.0
    if event_category(str(event_name)) in ("field", "relay", "other"):
        return 1.0
    d = get_event_dist(event_name)
    if d is None:
        return 1.0
    indoor = _is_indoor_sport(sport_name)
    f_len = track_length_factor_to_outdoor_flat(d, gender, lap_length_m=lap_length_m, indoor=indoor)
    f_bank = track_banking_factor_to_outdoor_flat(d, gender, banked=banked, indoor=indoor)
    return f_len * f_bank


def apply_track_venue_to_seconds(
    time_sec: float,
    event_name: str | None,
    gender: str,
    *,
    lap_length_m=None,
    banked=None,
    sport_name: str | None = None,
) -> float:
    """Apply $f_{\\mathrm{track}}$ for track results: $t \\leftarrow t \\cdot f_{\\mathrm{track}}$. """
    if time_sec is None or not math.isfinite(float(time_sec)) or time_sec <= 0:
        return time_sec
    f = track_venue_factor_to_outdoor_flat(
        event_name, gender, lap_length_m=lap_length_m, banked=banked, sport_name=sport_name
    )
    return float(time_sec) * f


_CD_LOOKUP_CACHE: dict[int, tuple[dict, dict]] = {}


def _course_details_lookups(course_details_df: pd.DataFrame) -> tuple[dict, dict]:
    """(meet, event, gender) -> row dict; fallback (meet, event) -> row dict."""
    cache_id = id(course_details_df)
    cached = _CD_LOOKUP_CACHE.get(cache_id)
    if cached is not None:
        return cached
    by_gender: dict[tuple, dict] = {}
    by_event: dict[tuple, dict] = {}
    for rec in course_details_df.to_dict("records"):
        mid = rec.get("meet_id")
        eid = rec.get("running_event_id")
        gender = rec.get("gender")
        if mid is None or eid is None:
            continue
        by_gender.setdefault((mid, eid, gender), rec)
        by_event.setdefault((mid, eid), rec)
    _CD_LOOKUP_CACHE[cache_id] = (by_gender, by_event)
    return by_gender, by_event


def get_course_details(row, course_details_df):
    by_gender, by_event = _course_details_lookups(course_details_df)
    mid = row["meet_id"]
    eid = row["running_event_id"]
    gender = row.get("gender")
    hit = by_gender.get((mid, eid, gender))
    if hit is not None:
        return hit
    hit = by_event.get((mid, eid))
    return hit if hit is not None else {}


def adjust_time_for_race(
    event_name: str,
    time: str,
    course_details: dict,
    gender: str,
    meet_elevation_feet=None,
    *,
    lap_length_m=None,
    banked=None,
    sport_name: str | None = None,
    wind_mps=None,
):
    time = parse_time(time)
    if pd.isna(event_name) or event_name is None:
        return time
    event_name = str(event_name)
    time = apply_wind_to_seconds(time, event_name, gender, wind_mps, sport_name=sport_name)
    if pd.notna(course_details.get("temperature")) and pd.notna(course_details.get("dew_point")):
        weather_factor = course_details["temperature"] + course_details["dew_point"]
        if weather_factor > 100:
            percent_increase = HEAT_QUADRATIC_COEFF * (weather_factor - 100) ** 2
            time *= 1 - (percent_increase / 100)
    time = apply_track_venue_to_seconds(
        time, event_name, gender, lap_length_m=lap_length_m, banked=banked, sport_name=sport_name
    )
    pb_hpa = barometric_pressure_hpa_from_record(course_details)
    time = apply_altitude_to_seconds(
        time,
        event_name,
        meet_elevation_feet,
        gender,
        barometric_pressure_hpa=pb_hpa,
    )
    return time


def get_event_dist(event_name):
    """Event distance in meters (alias for :func:`events.parse_event_distance_m`)."""
    return parse_event_distance_m(event_name)
