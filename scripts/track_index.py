"""NCAA facility indexing multipliers (2012--13) for track venue standardization.

Tables follow NCAA Indoor Track Size Conversion Charts (ncaa2012indoorconversion in one.bib);
see also Barnes & Malcata (2017) and Corts (2017). Factors are tabulated by event distance D (m)
and gender; NRCD applies them as f_len and f_bank in the Equation (general) product in cikm.tex.
"""

from __future__ import annotations

from typing import Literal

IndexKind = Literal["flat_to_banked", "undersized_to_flat"]

# (event_distance_m, multiplier) rows from NCAA / USTFCCCA charts; mile -> 1609 m.
_FLAT_TO_BANKED: dict[str, tuple[tuple[int, float], ...]] = {
    "M": (
        (200, 0.9824),
        (300, 0.9835),
        (400, 0.9843),
        (500, 0.9848),
        (600, 0.9852),
        (800, 0.9859),
        (1000, 0.9864),
        (1500, 0.9872),
        (1609, 0.9874),
        (3000, 0.9885),
        (5000, 0.9894),
    ),
    "F": (
        (200, 0.9847),
        (300, 0.9860),
        (400, 0.9869),
        (500, 0.9874),
        (600, 0.9879),
        (800, 0.9886),
        (1000, 0.9892),
        (1500, 0.9901),
        (1609, 0.9902),
        (3000, 0.9915),
        (5000, 0.9924),
    ),
}
_UNDERSIZED_TO_FLAT: dict[str, tuple[tuple[int, float], ...]] = {
    "M": (
        (200, 0.9872),
        (400, 0.9901),
        (800, 0.9923),
        (1000, 0.9929),
        (1609, 0.9941),
        (3000, 0.9953),
        (5000, 0.9961),
    ),
    "F": (
        (200, 0.9900),
        (400, 0.9929),
        (800, 0.9951),
        (1000, 0.9958),
        (1609, 0.9969),
        (3000, 0.9981),
        (5000, 0.9989),
    ),
}


def _gender_key(gender: str) -> str:
    return "F" if str(gender).upper() == "F" else "M"


def _nearest_row(distance_m: float, rows: tuple[tuple[int, float], ...]) -> float:
    key = min(rows, key=lambda row: abs(row[0] - distance_m))
    return key[1]


def ncaa_index_multiplier(
    event_distance_m: float,
    gender: str,
    kind: IndexKind,
) -> float:
    """Tabulated NCAA multiplier alpha(D) for event distance D (nearest standard distance)."""
    g = _gender_key(gender)
    if kind == "flat_to_banked":
        return _nearest_row(event_distance_m, _FLAT_TO_BANKED[g])
    return _nearest_row(event_distance_m, _UNDERSIZED_TO_FLAT[g])


def oversized_to_flat_multiplier(event_distance_m: float, gender: str) -> float:
    """alpha_ot(D) = 1 / alpha_fb(D); NCAA treats banked and oversized with the same indexing."""
    c_fb = ncaa_index_multiplier(event_distance_m, gender, "flat_to_banked")
    return 1.0 / c_fb
