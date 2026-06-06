"""Coverage by era (before/after Aug 2023) and metadata availability."""

from __future__ import annotations

import pandas as pd

from config import EVENT_CATEGORIES, SPORTS


def coverage_by_sport_era(df: pd.DataFrame) -> list[dict]:
    """Per sport and era: counts and metadata coverage (weather, altitude, course features)."""
    rows = []
    for sport in SPORTS:
        sub = df[df["sport_name"] == sport]
        for era in ("historical", "comprehensive"):
            e = sub[sub["era"] == era]
            if len(e) == 0:
                rows.append(
                    {
                        "sport": sport,
                        "era": era,
                        "results": 0,
                        "athletes": 0,
                        "meets": 0,
                        "weather_pct": 0.0,
                        "barometric_pct": 0.0,
                        "course_features_pct": 0.0,
                        "meet_altitude_pct": 0.0,
                        "metadata_pct": 0.0,
                    }
                )
                continue
            weather_pct = round(100 * e["has_weather_meta"].mean(), 1)
            barometric_pct = round(100 * e["has_barometric_meta"].mean(), 1)
            course_features_pct = round(100 * e["has_course_features_meta"].mean(), 1)
            meet_altitude_pct = round(100 * e["has_meet_altitude"].mean(), 1)
            rows.append(
                {
                    "sport": sport,
                    "era": era,
                    "results": int(len(e)),
                    "athletes": int(e["athlete_id"].nunique()),
                    "meets": int(e["meet_id"].nunique()),
                    "weather_pct": weather_pct,
                    "barometric_pct": barometric_pct,
                    "course_features_pct": course_features_pct,
                    "meet_altitude_pct": meet_altitude_pct,
                    "metadata_pct": weather_pct,
                }
            )
    return rows


def gender_by_sport_era(df: pd.DataFrame) -> list[dict]:
    rows = []
    for sport in SPORTS:
        for era in ("historical", "comprehensive"):
            for gender in ("F", "M"):
                e = df[(df["sport_name"] == sport) & (df["era"] == era) & (df["gender"] == gender)]
                if len(e) == 0:
                    continue
                rows.append(
                    {
                        "sport": sport,
                        "era": era,
                        "gender": gender,
                        "results": int(len(e)),
                        "metadata_pct": round(100 * e["has_weather_meta"].mean(), 1),
                    }
                )
    return rows
