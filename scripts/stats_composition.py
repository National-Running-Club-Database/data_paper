"""Dataset scale and sport-level composition statistics."""

from __future__ import annotations

import pandas as pd

from itertools import combinations

from config import SPORTS
from load_data import build_results_frame


def athlete_sport_overlap(df: pd.DataFrame) -> dict:
    """Distinct athletes per sport overlap (sport-level counts are not additive)."""
    sport_sets = df.groupby("athlete_id")["sport_name"].apply(lambda x: frozenset(x.unique()))
    n = int(len(sport_sets))
    by_n_sports = sport_sets.apply(len).value_counts().sort_index()
    multi = int((sport_sets.apply(len) > 1).sum())
    per_sport_sum = int(df.groupby("sport_name")["athlete_id"].nunique().sum())
    pairs = {}
    for a, b in combinations(SPORTS, 2):
        pairs[f"{a}|{b}"] = int(sport_sets.apply(lambda s, a=a, b=b: a in s and b in s).sum())
    return {
        "athletes_with_results": n,
        "per_sport_athlete_sum": per_sport_sum,
        "single_sport": int((sport_sets.apply(len) == 1).sum()),
        "multi_sport": multi,
        "multi_sport_pct": round(100.0 * multi / n, 1) if n else None,
        "three_plus_sports": int((sport_sets.apply(len) >= 3).sum()),
        "four_sports": int((sport_sets.apply(len) == 4).sum()),
        "by_num_sports": {int(k): int(v) for k, v in by_n_sports.items()},
        "pairwise": pairs,
    }


def composition_summary(tables: dict, df: pd.DataFrame) -> dict:
    result = tables["result"]
    meet = tables["meet"]
    athlete = tables["athlete"]

    meet_dates = pd.to_datetime(meet["start_date"], errors="coerce")
    counts = (
        df.groupby("sport_name")
        .agg(results=("result_id", "count"), athletes=("athlete_id", "nunique"), meets=("meet_id", "nunique"))
        .reset_index()
    )
    by_sport = []
    for name in SPORTS:
        row = counts[counts["sport_name"] == name]
        if len(row):
            by_sport.append(row.iloc[0].to_dict())
        else:
            by_sport.append({"sport_name": name, "results": 0, "athletes": 0, "meets": 0})

    era_counts = df.groupby("era").size().to_dict()

    return {
        "total_results": int(len(result)),
        "total_athletes": int(df["athlete_id"].nunique()),
        "roster_athletes": int(athlete["athlete_id"].nunique()),
        "total_meets": int(meet["meet_id"].nunique()),
        "total_teams": int(tables["team"]["team_id"].nunique()),
        "course_detail_records": int(len(tables["course_details"])),
        "athlete_team_associations": int(len(tables["athlete_team_association"])),
        "by_sport": by_sport,
        "results_by_era": {k: int(v) for k, v in era_counts.items()},
        "gender_athletes": {
            k: int(v) for k, v in df.groupby("gender")["athlete_id"].nunique().items()
        },
        "date_min": str(meet_dates.min().date()) if meet_dates.notna().any() else None,
        "date_max": str(meet_dates.max().date()) if meet_dates.notna().any() else None,
        "athlete_sport_overlap": athlete_sport_overlap(df),
    }
