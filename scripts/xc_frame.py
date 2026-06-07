"""Shared XC result frame: merge tables once and attach course metadata."""

from __future__ import annotations

import pandas as pd

from events import event_category
from schema import meet_altitude_column

_COURSE_DETAIL_COLS = (
    "temperature",
    "dew_point",
    "elevation_gain",
    "elevation_loss",
    "estimated_course_distance",
    "barometric_pressure",
)


def prepare_xc_results(
    tables: dict,
    *,
    exclude_nationals: bool = True,
) -> pd.DataFrame:
    """XC results merged with athlete, event, meet altitude; optional nationals filter."""
    meet = tables["meet"]
    result = tables["result"]
    athlete = tables["athlete"]
    running_event = tables["running_event"]

    xc_meets = set(meet.loc[meet["sport_id"] == 1, "meet_id"])
    df = result[result["meet_id"].isin(xc_meets)].copy()
    df = df.merge(athlete[["athlete_id", "gender"]], on="athlete_id")
    df = df.merge(running_event, on="running_event_id")
    df["event_category"] = df["event_name"].apply(event_category)

    alt_col = meet_altitude_column(meet)
    df = df.merge(
        meet[["meet_id", "start_date", alt_col]].rename(columns={alt_col: "altitude"}),
        on="meet_id",
    )
    df["race_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["season"] = df["race_date"].dt.year

    if exclude_nationals:
        nationals = meet.set_index("meet_id")["nationals"].astype(bool)
        df = df[~df["meet_id"].map(nationals).fillna(False)]

    return df


def attach_course_details(df: pd.DataFrame, course_details_df: pd.DataFrame) -> pd.DataFrame:
    """Join course metadata; gender-specific match with event-level fallback."""
    present = [c for c in _COURSE_DETAIL_COLS if c in course_details_df.columns]
    if not present:
        return df

    cd = course_details_df[["meet_id", "running_event_id", "gender", *present]].copy()
    cd = cd.rename(columns={c: f"cd_{c}" for c in present})
    out = df.merge(cd, on=["meet_id", "running_event_id", "gender"], how="left")

    fb = course_details_df[["meet_id", "running_event_id", *present]].drop_duplicates(
        ["meet_id", "running_event_id"], keep="first"
    )
    fb = fb.rename(columns={c: f"cd_fb_{c}" for c in present})
    out = out.merge(fb, on=["meet_id", "running_event_id"], how="left")
    for col in present:
        out[f"cd_{col}"] = out[f"cd_{col}"].fillna(out[f"cd_fb_{col}"])
        out.drop(columns=[f"cd_fb_{col}"], inplace=True)
    return out


def build_xc_frame(
    tables: dict,
    *,
    exclude_nationals: bool = True,
) -> pd.DataFrame:
    """Prepared XC rows with course-detail columns attached."""
    return attach_course_details(
        prepare_xc_results(tables, exclude_nationals=exclude_nationals),
        tables["course_details"],
    )
