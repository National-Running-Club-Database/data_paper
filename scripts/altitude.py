"""Peronnet–Thibault altitude standardization (meet elevation in feet)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any, Literal

from events import applies_altitude_conversion, parse_event_distance_m

VenueElevationUnit = Literal["ft", "m"]

_PERONNET_BMR = 1.2
_PERONNET_K1 = 30.0
_PERONNET_K2 = 20.0
_PERONNET_TMAP = 420.0
_PERONNET_F_ANAEROBIC = 0.233
_PERONNET_RHO_SEA = 1.204
_PERONNET_AD_OVER_K = {"M": 17e-3, "F": 21e-3}
_PERONNET_PARAMS = {
    "M": {"A": 1669.0, "MAP": 29.2, "E": 14.0},
    "F": {"A": 1575.0, "MAP": 26.4, "E": 13.5},
}


def _peronnet_gender(gender: str) -> str:
    return "F" if str(gender).upper() == "F" else "M"


def venue_elevation_to_feet(value: float | None, unit: VenueElevationUnit = "ft") -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v * 3.280839895 if unit == "m" else v


def _peronnet_barometric_pressure_torr(elevation_m: float) -> float:
    return 760.0 * (1.0 - elevation_m / 44330.8) ** 5.255


def altitude_power_percent_sea_level(elevation_ft: float) -> float:
    elevation_m = elevation_ft * 0.3048
    pb = _peronnet_barometric_pressure_torr(elevation_m)
    return (
        -174.1448622
        + 1.0899959 * pb
        - 1.5119e-3 * pb**2
        + 0.72674e-6 * pb**3
    )


def _peronnet_map_ratio(elevation_ft: float) -> float:
    return altitude_power_percent_sea_level(elevation_ft) / 100.0


def _peronnet_avg_power(
    duration_s: float, map_wkg: float, anaerobic_jkg: float, endurance_e: float
) -> float:
    if duration_s <= 0:
        return 0.0
    if duration_s < _PERONNET_TMAP:
        s = anaerobic_jkg
        b = map_wkg - _PERONNET_BMR
    else:
        s = anaerobic_jkg - _PERONNET_F_ANAEROBIC * anaerobic_jkg * math.log(
            duration_s / _PERONNET_TMAP
        )
        b = map_wkg - _PERONNET_BMR + endurance_e * math.log(duration_s / _PERONNET_TMAP)
    aerobic = _PERONNET_BMR + b * (
        1.0 - (_PERONNET_K1 / duration_s) * (1.0 - math.exp(-duration_s / _PERONNET_K1))
    )
    anaerobic = (s / duration_s) * (1.0 - math.exp(-duration_s / _PERONNET_K2))
    return aerobic + anaerobic


def _peronnet_running_power_required(
    speed_mps: float, distance_m: float, air_density: float, gender: str
) -> float:
    ad_over_k = _PERONNET_AD_OVER_K[_peronnet_gender(gender)]
    return (
        _PERONNET_BMR
        + 3.86 * speed_mps
        + 0.5 * air_density * ad_over_k * speed_mps**3
        + (2.0 * speed_mps**3) / distance_m
    )


def _peronnet_predict_race_time(
    distance_m: float,
    map_wkg: float,
    gender: str,
    air_density: float | None = None,
) -> float | None:
    if distance_m <= 0:
        return None
    if air_density is None:
        air_density = _PERONNET_RHO_SEA
    params = _PERONNET_PARAMS[_peronnet_gender(gender)]
    lo = max(1.0, distance_m / 15.0)
    hi = max(lo + 1.0, distance_m * 0.35)
    if distance_m >= 3000:
        hi = max(hi, 7200.0)
    for _ in range(100):
        mid = (lo + hi) / 2.0
        speed = distance_m / mid
        if _peronnet_avg_power(mid, map_wkg, params["A"], params["E"]) >= _peronnet_running_power_required(
            speed, distance_m, air_density, gender
        ):
            hi = mid
        else:
            lo = mid
    return hi


def barometric_pressure_hpa_from_record(record: Mapping[str, Any] | None) -> float | None:
    if not record:
        return None
    for key in ("barometric_pressure", "barometric_pressure_hpa"):
        if key not in record:
            continue
        parsed = parse_barometric_pressure_hpa(record.get(key))
        if parsed is not None:
            return parsed
    return None


def parse_barometric_pressure_hpa(pressure_hpa: float | None) -> float | None:
    if pressure_hpa is None:
        return None
    try:
        hpa = float(pressure_hpa)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(hpa) or hpa <= 0:
        return None
    return hpa


def barometric_pressure_torr_from_hpa(pressure_hpa: float | None) -> float | None:
    hpa = parse_barometric_pressure_hpa(pressure_hpa)
    if hpa is None:
        return None
    return hpa * 0.750062


def resolve_meet_altitude_inputs(
    meet_elevation: float | None,
    barometric_pressure_hpa: float | None = None,
    *,
    elevation_unit: VenueElevationUnit = "ft",
    warn_on_orphan_pressure: bool = True,
    stacklevel: int = 3,
) -> tuple[float | None, float | None]:
    elev_ft = venue_elevation_to_feet(meet_elevation, elevation_unit)
    pb_hpa = parse_barometric_pressure_hpa(barometric_pressure_hpa)
    if elev_ft is None:
        if warn_on_orphan_pressure and pb_hpa is not None:
            warnings.warn(
                "barometric_pressure ignored without meet_elevation: "
                "altitude correction requires venue elevation (MAP); "
                "race-time pressure alone is not applied.",
                UserWarning,
                stacklevel=stacklevel,
            )
        return None, None
    return elev_ft, pb_hpa


def peronnet_f_alt(
    distance_m: float,
    elevation_ft: float | None,
    gender: str = "M",
    *,
    barometric_pressure_torr: float | None = None,
) -> float:
    if distance_m <= 0:
        return 1.0
    z = venue_elevation_to_feet(elevation_ft, "ft")
    if z is None:
        return 1.0

    gender_key = str(gender).upper()
    if gender_key == "MIXED":
        return (
            peronnet_f_alt(
                distance_m, z, "M", barometric_pressure_torr=barometric_pressure_torr
            )
            + peronnet_f_alt(
                distance_m, z, "F", barometric_pressure_torr=barometric_pressure_torr
            )
        ) / 2.0

    params = _PERONNET_PARAMS[_peronnet_gender(gender)]
    elevation_m = z * 0.3048
    map_alt = params["MAP"] * _peronnet_map_ratio(z)
    pb_race = barometric_pressure_torr
    if pb_race is None or not math.isfinite(pb_race) or pb_race <= 0:
        pb_race = _peronnet_barometric_pressure_torr(elevation_m)
    rho_alt = _PERONNET_RHO_SEA * (pb_race / 760.0)

    t_sea = _peronnet_predict_race_time(distance_m, params["MAP"], gender, _PERONNET_RHO_SEA)
    t_alt = _peronnet_predict_race_time(distance_m, map_alt, gender, rho_alt)
    if not t_sea or not t_alt or t_alt <= 0:
        return 1.0
    return float(t_sea / t_alt)


def sea_level_time_seconds(
    time_sec: float,
    distance_m: float,
    elevation_ft: float | None,
    gender: str = "M",
    *,
    barometric_pressure_hpa: float | None = None,
) -> float:
    if not math.isfinite(time_sec) or time_sec <= 0 or distance_m <= 0:
        return time_sec
    pb_torr = barometric_pressure_torr_from_hpa(barometric_pressure_hpa)
    return float(time_sec) * peronnet_f_alt(
        distance_m, elevation_ft, gender, barometric_pressure_torr=pb_torr
    )


def apply_meet_altitude(
    time_sec: float,
    event_name: str | None,
    meet_elevation: float | None,
    gender: str,
    *,
    elevation_unit: VenueElevationUnit = "ft",
    barometric_pressure_hpa: float | None = None,
    warn_on_orphan_pressure: bool = True,
) -> float:
    if not applies_altitude_conversion(event_name):
        return time_sec
    dist = parse_event_distance_m(event_name)
    if dist is None:
        return time_sec
    elev_ft, pb_hpa = resolve_meet_altitude_inputs(
        meet_elevation,
        barometric_pressure_hpa,
        elevation_unit=elevation_unit,
        warn_on_orphan_pressure=warn_on_orphan_pressure,
        stacklevel=4,
    )
    if elev_ft is None:
        return time_sec
    pb_torr = barometric_pressure_torr_from_hpa(pb_hpa)
    f = peronnet_f_alt(dist, elev_ft, gender, barometric_pressure_torr=pb_torr)
    return float(time_sec) * f
