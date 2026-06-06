"""Event-specific sprint wind corrections (no linear D/100 scaling on curved tracks).

Tabulated calm-equivalent seconds at +/-2 m/s (linear in w below cap), by event bucket:
- 100 m: Linthorne (1994) empirical adjustments (+2 tail / -2 head).
- 200 m: Quinn (2003) Table 2, straight-gauge reading (effective ~0.95 m/s tail at +2 gauge).
- 110 m hurdles: Spiegel & Mureika (2003) model (~0.19 s M at +2 m/s tail).
- 400 m: Quinn (2004) curved-loop model; gauge is straight-component only and net effect
  varies with wind direction---values here are conservative down-straight gauge estimates,
  not 4x the 100 m correction.
"""

from __future__ import annotations

# (event_distance_m, gender) -> seconds to ADD to raw time for calm equivalent at w=+2 m/s (tailwind)
_TAIL_GAIN_AT_2MPS: dict[tuple[int, str], float] = {
    (100, "M"): 0.101,
    (100, "F"): 0.110,
    (110, "M"): 0.190,
    (110, "F"): 0.200,
    (200, "M"): 0.112,
    (200, "F"): 0.123,
    (400, "M"): 0.090,
    (400, "F"): 0.100,
}

# seconds to ADD at w=-2 m/s (headwind penalty)
_HEAD_LOSS_AT_2MPS: dict[tuple[int, str], float] = {
    (100, "M"): 0.121,
    (100, "F"): 0.134,
    (110, "M"): 0.125,
    (110, "F"): 0.138,
    (200, "M"): 0.121,
    (200, "F"): 0.135,
    (400, "M"): 0.130,
    (400, "F"): 0.140,
}

_WIND_BUCKETS_M = (100, 110, 200, 400)


def _gender_key(gender: str) -> str:
    return "F" if str(gender).upper() == "F" else "M"


def wind_event_bucket_m(event_distance_m: float) -> int:
    """Map race distance to published correction row (100 / 110 / 200 / 400 m)."""
    d = float(event_distance_m)
    if d <= 105:
        return 100
    if d <= 115:
        return 110
    if d <= 205:
        return 200
    return 400


def wind_delta_seconds_at_speed(
    wind_mps: float,
    event_distance_m: float,
    gender: str,
) -> float:
    """Calm-equivalent shift Delta t for recorded wind (m/s); linear in |w| below +/-2 m/s cap."""
    if event_distance_m is None or event_distance_m <= 0:
        return 0.0

    bucket = wind_event_bucket_m(event_distance_m)
    g = _gender_key(gender)
    w = float(wind_mps)
    cap = 2.0
    if abs(w) > cap:
        w = cap if w > 0 else -cap

    if w >= 0:
        delta_2 = _TAIL_GAIN_AT_2MPS[(bucket, g)]
        return delta_2 * (w / 2.0)
    delta_2 = _HEAD_LOSS_AT_2MPS[(bucket, g)]
    return delta_2 * (abs(w) / 2.0)
