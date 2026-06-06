"""Longitudinal depth: athletes appearing in multiple school years (Aug 1 boundary)."""

from __future__ import annotations

import pandas as pd

# US collegiate running year: Aug 1 through following July 31.
_SCHOOL_YEAR_START_MONTH = 8


def school_year_start(dates: pd.Series) -> pd.Series:
    """Label each meet date by school-year start (e.g. 2023-08-01 -> 2023)."""
    d = pd.to_datetime(dates, errors="coerce")
    return d.dt.year.where(d.dt.month >= _SCHOOL_YEAR_START_MONTH, d.dt.year - 1)


def longitudinal_depth_summary(df: pd.DataFrame) -> dict:
    """Count distinct athletes by number of school years with approved results."""
    seasons = df.groupby("athlete_id")["meet_date"].apply(
        lambda s: school_year_start(s.dropna()).nunique()
    )
    n = int(len(seasons))
    thresholds = [1, 2, 3, 4, 5]
    by_min_seasons = []
    for t in thresholds:
        count = int((seasons >= t).sum())
        by_min_seasons.append(
            {
                "min_seasons": t,
                "n_athletes": count,
                "pct_of_athletes": round(100.0 * count / n, 1) if n else None,
            }
        )
    return {
        "total_athletes_with_results": n,
        "by_min_seasons": by_min_seasons,
    }
