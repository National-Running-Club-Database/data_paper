"""Shared paths and constants for NRCD analysis scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

COMPREHENSIVE_FROM = "2023-08-01"

SPORTS = [
    "Cross Country",
    "Indoor Track",
    "Outdoor Track",
    "Road Race",
]

EVENT_CATEGORIES = ["sprint", "mid_distance", "distance", "relay", "field"]

RIEGEL_B_MEN = 1.055  # noqa: N816  # Riegel (1981) table, rounded
RIEGEL_B_WOMEN = 1.08  # Riegel (1981) table; unified 1.06 in blythe2016prediction baseline
XC_TARGET_M = 8000.0
XC_TARGET_F = 6000.0

# Quadratic fit to Hadley temp+dew bands: slowdown (%) = k * (H - 100)^2, f = 1 - slowdown/100
# Rounded from least-squares fit (0.001642) to Hadley band midpoints; see heat_compare_k.py
HEAT_QUADRATIC_COEFF = 0.0016

# Peronnet-Thibault altitude: apply to running events with known distance (incl. sprints)

# Track venue reference: outdoor 400 m flat lap (NCAA facility indexing; see utils.track_*_factor)
TRACK_OUTDOOR_REFERENCE_LAP_M = 400.0

# Wind: event-specific deltas in wind_index.py (Quinn 2003/2004, Linthorne 1994, Mureika 2008)
WIND_MAX_APPLY_MPS = 4.0
