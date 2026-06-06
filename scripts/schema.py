"""NRCD CSV column helpers for analysis scripts."""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd


def meet_altitude_column(df: pd.DataFrame) -> str:
    """Return meet-table altitude column name (``altitude`` or legacy ``elevation``)."""
    if "altitude" in df.columns:
        return "altitude"
    if "elevation" in df.columns:
        return "elevation"
    raise KeyError("meet table missing altitude/elevation column")


def _finite_positive_ft(value: Any) -> float | None:
    if value is None:
        return None
    try:
        z = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(z) or z <= 0:
        return None
    return z


def meet_altitude_ft_from_record(
    row: Mapping[str, Any] | Any,
    course_details: Mapping[str, Any] | None = None,
) -> float | None:
    """Meet venue altitude (ft) from merged result row or ``course_details``."""
    elev = None
    if hasattr(row, "get"):
        elev = row.get("altitude")
        if elev is None or (isinstance(elev, float) and pd.isna(elev)):
            elev = row.get("elevation")
    if elev is None or (isinstance(elev, float) and pd.isna(elev)):
        if course_details:
            elev = course_details.get("altitude") or course_details.get("meet_elevation")
            if elev is None:
                elev = course_details.get("elevation")
    return _finite_positive_ft(elev)
