"""
Hadley (Maximum Performance Running) piecewise heat bands and quadratic surrogate.

Source: temp (°F) + dew point (°F); adjustment ranges from coaching table
https://maximumperformancerunning.blogspot.com/2013/07/temperature-dew-point.html
"""

from __future__ import annotations

import numpy as np

from config import HEAT_QUADRATIC_COEFF
from standardization import weather_factor

# (H_min, H_max, slowdown_pct_lo, slowdown_pct_hi) for H = temp + dew (°F)
HADLEY_BANDS: list[tuple[int, int, float, float]] = [
    (0, 100, 0.0, 0.0),
    (101, 110, 0.0, 0.5),
    (111, 120, 0.5, 1.0),
    (121, 130, 1.0, 2.0),
    (131, 140, 2.0, 3.0),
    (141, 150, 3.0, 4.5),
    (151, 160, 4.5, 6.0),
    (161, 170, 6.0, 8.0),
    (171, 180, 8.0, 10.0),
]


def hadley_band(h: float) -> tuple[int, int, float, float] | None:
    """Return (h_min, h_max, pct_lo, pct_hi) for heat index h, or None if h > 180."""
    for h_min, h_max, lo, hi in HADLEY_BANDS:
        if h_min <= h <= h_max:
            return h_min, h_max, lo, hi
    if h > 180:
        return 181, 999, 10.0, 10.0  # table: hard running not recommended; cap at 10%
    return None


def hadley_slowdown_pct_midpoint(h: float) -> float:
    """Midpoint of published slowdown range for heat index h (percent)."""
    band = hadley_band(h)
    if band is None:
        return 0.0
    _, _, lo, hi = band
    return (lo + hi) / 2.0


def hadley_slowdown_pct_linear(h: float) -> float:
    """Linear interpolation within the Hadley band (percent)."""
    if h <= 100:
        return 0.0
    band = hadley_band(h)
    if band is None:
        return 10.0
    h_min, h_max, lo, hi = band
    if h_max <= h_min:
        return lo
    t = (h - h_min) / (h_max - h_min)
    return lo + t * (hi - lo)


def quadratic_slowdown_pct(h: float, k: float = HEAT_QUADRATIC_COEFF) -> float:
    """NRCD quadratic surrogate: percent slowdown before dividing by 100."""
    if h <= 100:
        return 0.0
    return k * (h - 100) ** 2


def quadratic_weather_factor(h: float, k: float = HEAT_QUADRATIC_COEFF) -> float:
    pct = quadratic_slowdown_pct(h, k)
    return 1.0 - pct / 100.0


def fit_quadratic_coefficient() -> float:
    """Least-squares k in pct = k*(H-100)^2 vs Hadley band midpoints (101--180)."""
    hs, targets = [], []
    for h_min, h_max, lo, hi in HADLEY_BANDS:
        if h_max <= 100:
            continue
        h_mid = (h_min + h_max) / 2.0
        hs.append(h_mid)
        targets.append((lo + hi) / 2.0)
    x = np.array([(h - 100) ** 2 for h in hs], dtype=float)
    y = np.array(targets, dtype=float)
    return float(np.dot(x, y) / np.dot(x, x))


def validate_piecewise_to_quadratic() -> dict:
    """
    Validate that f = 1 - k*(H-100)^2/100 approximates Hadley piecewise bands.
    """
    k_deployed = HEAT_QUADRATIC_COEFF
    k_ls = fit_quadratic_coefficient()

    # Band midpoints 101--180
    mids = []
    for h_min, h_max, lo, hi in HADLEY_BANDS:
        if h_max <= 100:
            continue
        h_mid = (h_min + h_max) / 2.0
        quad = quadratic_slowdown_pct(h_mid, k_deployed)
        target = (lo + hi) / 2.0
        mids.append(
            {
                "H_mid": h_mid,
                "hadley_pct_lo": lo,
                "hadley_pct_hi": hi,
                "hadley_pct_mid": round(target, 3),
                "quadratic_pct": round(quad, 3),
                "error_pct": round(quad - target, 3),
            }
        )

    errors_mid = [m["error_pct"] for m in mids]
    rmse_mid = float(np.sqrt(np.mean(np.array(errors_mid) ** 2)))

    # Integer H in 101--180: share within Hadley band; max deviation from band
    in_band = 0
    n_hot = 0
    max_below_lo = 0.0
    max_above_hi = 0.0
    for h in range(101, 181):
        band = hadley_band(h)
        if not band:
            continue
        _, _, lo, hi = band
        q = quadratic_slowdown_pct(h, k_deployed)
        n_hot += 1
        if lo <= q <= hi:
            in_band += 1
        if q < lo:
            max_below_lo = max(max_below_lo, lo - q)
        if q > hi:
            max_above_hi = max(max_above_hi, q - hi)

    # Compare deployed k vs LS on same grid (linear reference)
    hs_lin = np.arange(101, 181, dtype=float)
    ref = np.array([hadley_slowdown_pct_linear(h) for h in hs_lin])
    quad_deployed = np.array([quadratic_slowdown_pct(h, k_deployed) for h in hs_lin])
    quad_ls = np.array([quadratic_slowdown_pct(h, k_ls) for h in hs_lin])

    return {
        "hadley_source": "Maximum Performance Running temp+dew bands (2013)",
        "deployed_coefficient_k": k_deployed,
        "least_squares_k_vs_band_midpoints": round(k_ls, 6),
        "formula": "slowdown_pct = k * (H - 100)^2; f_weather = 1 - slowdown_pct / 100",
        "band_midpoint_comparison": mids,
        "rmse_pct_vs_band_midpoints": round(rmse_mid, 4),
        "integer_H_101_to_180": {
            "n": n_hot,
            "pct_within_hadley_band": round(100 * in_band / n_hot, 1) if n_hot else None,
            "max_pct_below_band_lo": round(max_below_lo, 4),
            "max_pct_above_band_hi": round(max_above_hi, 4),
        },
        "rmse_pct_vs_linear_piecewise_101_180": {
            "deployed_k": round(float(np.sqrt(np.mean((quad_deployed - ref) ** 2))), 4),
            "least_squares_k": round(float(np.sqrt(np.mean((quad_ls - ref) ** 2))), 4),
        },
        "example_factors": {
            "H_110": round(weather_factor(55, 55), 6),
            "H_145": round(weather_factor(72.5, 72.5), 6),
            "H_160": round(weather_factor(80, 80), 6),
        },
        "validation_passed": bool(
            rmse_mid < 1.0
            and (in_band / n_hot >= 0.5 if n_hot else False)
            and abs(k_ls - k_deployed) / k_ls < 0.15
        ),
    }
