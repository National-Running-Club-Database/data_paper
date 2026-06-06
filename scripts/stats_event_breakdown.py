"""Result counts by sport, gender, event category, and era."""

from __future__ import annotations

import pandas as pd

from config import EVENT_CATEGORIES, SPORTS


def event_breakdown(df: pd.DataFrame) -> dict:
    """Tables for paper: by sport×gender×category and sport×era×gender×category."""
    by_sport_gender_category = []
    for sport in SPORTS:
        for gender in ("F", "M"):
            base = df[(df["sport_name"] == sport) & (df["gender"] == gender)]
            row = {
                "sport": sport,
                "gender": gender,
                "results": int(len(base)),
            }
            for cat in EVENT_CATEGORIES:
                row[cat] = int((base["event_category"] == cat).sum())
            by_sport_gender_category.append(row)

    by_sport_era_gender_category = []
    for sport in SPORTS:
        for era in ("historical", "comprehensive"):
            for gender in ("F", "M"):
                base = df[
                    (df["sport_name"] == sport)
                    & (df["era"] == era)
                    & (df["gender"] == gender)
                ]
                if len(base) == 0:
                    continue
                row = {
                    "sport": sport,
                    "era": era,
                    "gender": gender,
                    "results": int(len(base)),
                    "metadata_pct": round(100 * base["has_weather_meta"].mean(), 1),
                }
                for cat in EVENT_CATEGORIES:
                    row[cat] = int((base["event_category"] == cat).sum())
                by_sport_era_gender_category.append(row)

    totals_by_category = {
        cat: int((df["event_category"] == cat).sum()) for cat in EVENT_CATEGORIES
    }

    return {
        "by_sport_gender_category": by_sport_gender_category,
        "by_sport_era_gender_category": by_sport_era_gender_category,
        "totals_by_category": totals_by_category,
    }
