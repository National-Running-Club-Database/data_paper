"""Load CSV tables and build an analysis-ready results frame."""

from __future__ import annotations

import pandas as pd

from config import COMPREHENSIVE_FROM, DATA
from events import event_category

from schema import meet_altitude_column


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "result": pd.read_csv(DATA / "result.csv", low_memory=False),
        "meet": pd.read_csv(DATA / "meet.csv"),
        "athlete": pd.read_csv(DATA / "athlete.csv"),
        "sport": pd.read_csv(DATA / "sport.csv"),
        "running_event": pd.read_csv(DATA / "running_event.csv"),
        "course_details": pd.read_csv(DATA / "course_details.csv"),
        "team": pd.read_csv(DATA / "team.csv"),
        "athlete_team_association": pd.read_csv(DATA / "athlete_team_association.csv"),
    }


def build_results_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merged approved results with sport, gender, era, event category, metadata flags."""
    result = tables["result"]
    meet = tables["meet"]
    athlete = tables["athlete"]
    sport = tables["sport"]
    running_event = tables["running_event"]
    course_details = tables["course_details"]

    alt_col = meet_altitude_column(meet)
    df = result.merge(
        meet[["meet_id", "sport_id", "start_date", alt_col]].rename(columns={alt_col: "altitude"}),
        on="meet_id",
    )
    df = df.merge(sport, on="sport_id")
    df = df.merge(athlete[["athlete_id", "gender"]], on="athlete_id")
    df = df.merge(running_event, on="running_event_id")

    df["meet_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    cut = pd.Timestamp(COMPREHENSIVE_FROM)
    df["era"] = df["meet_date"].apply(
        lambda d: "comprehensive" if pd.notna(d) and d >= cut else "historical"
    )

    cd_weather = course_details.dropna(subset=["temperature", "dew_point"])
    cd_keys = cd_weather[["meet_id", "running_event_id", "gender"]].drop_duplicates()
    df = df.merge(
        cd_keys.assign(has_weather_meta=1),
        on=["meet_id", "running_event_id", "gender"],
        how="left",
    )
    df["has_weather_meta"] = df["has_weather_meta"].fillna(0).astype(bool)

    baro_col = (
        "barometric_pressure"
        if "barometric_pressure" in course_details.columns
        else "barometric_pressure_hpa"
    )
    if baro_col in course_details.columns:
        cd_baro = course_details.dropna(subset=[baro_col])
        baro_keys = cd_baro[["meet_id", "running_event_id", "gender"]].drop_duplicates()
        df = df.merge(
            baro_keys.assign(has_barometric_meta=1),
            on=["meet_id", "running_event_id", "gender"],
            how="left",
        )
    if "has_barometric_meta" not in df.columns:
        df["has_barometric_meta"] = False
    else:
        df["has_barometric_meta"] = df["has_barometric_meta"].fillna(0).astype(bool)

    df["has_meet_altitude"] = df["altitude"].notna() & (df["altitude"] > 0)

    cf_cols = ("elevation_gain", "elevation_loss", "estimated_course_distance")
    if any(c in course_details.columns for c in cf_cols):
        mask = pd.Series(False, index=course_details.index)
        for col in cf_cols:
            if col in course_details.columns:
                mask |= course_details[col].notna()
        cf_keys = course_details.loc[mask][
            ["meet_id", "running_event_id", "gender"]
        ].drop_duplicates()
        df = df.merge(
            cf_keys.assign(has_course_features_meta=1),
            on=["meet_id", "running_event_id", "gender"],
            how="left",
        )
    if "has_course_features_meta" not in df.columns:
        df["has_course_features_meta"] = False
    else:
        df["has_course_features_meta"] = df["has_course_features_meta"].fillna(0).astype(bool)

    df["event_category"] = df["event_name"].apply(event_category)

    return df
