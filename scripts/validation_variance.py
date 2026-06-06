"""Within-athlete cross-meet variance validation (cross country)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from standardization import converted_only_xc, standardize_xc_row
from utils import parse_time


def within_athlete_variance(tables: dict, min_races: int = 3) -> dict:
    meet = tables["meet"]
    result = tables["result"]
    athlete = tables["athlete"]
    running_event = tables["running_event"]
    course_details = tables["course_details"]

    xc_meets = set(meet.loc[meet["sport_id"] == 1, "meet_id"])
    df = result[result["meet_id"].isin(xc_meets)].copy()
    df = df.merge(athlete[["athlete_id", "gender"]], on="athlete_id")
    df = df.merge(running_event, on="running_event_id")
    from schema import meet_altitude_column

    alt_col = meet_altitude_column(meet)
    df = df.merge(
        meet[["meet_id", "start_date", alt_col]].rename(columns={alt_col: "altitude"}),
        on="meet_id",
    )
    df["season"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year
    nationals = meet.set_index("meet_id")["nationals"].astype(bool)
    df = df[~df["meet_id"].map(nationals).fillna(False)]

    records = []
    for _, row in df.iterrows():
        raw, std = standardize_xc_row(row, course_details)
        conv = converted_only_xc(row, course_details)
        if np.isfinite(raw) and np.isfinite(std):
            records.append(
                {
                    "athlete_id": row["athlete_id"],
                    "season": row["season"],
                    "meet_id": row["meet_id"],
                    "gender": row["gender"],
                    "raw": raw,
                    "standardized": std,
                    "converted_only": conv,
                }
            )

    perf = pd.DataFrame(records).dropna(subset=["season"])
    rows_out = []
    all_season_stats = []
    for gender in ["M", "F"]:
        g = perf[perf["gender"] == gender]
        athlete_season_stats = []
        for (_, _), grp in g.groupby(["athlete_id", "season"]):
            if grp["meet_id"].nunique() < min_races:
                continue
            athlete_season_stats.append(
                {
                    "raw_sd": grp["raw"].std(),
                    "std_sd": grp["standardized"].std(),
                    "conv_sd": grp["converted_only"].std(),
                }
            )
        all_season_stats.extend(athlete_season_stats)
        if not athlete_season_stats:
            continue
        stats_df = pd.DataFrame(athlete_season_stats)
        med_raw = stats_df["raw_sd"].median()
        med_conv = stats_df["conv_sd"].median()
        med_std = stats_df["std_sd"].median()
        rows_out.append(
            {
                "gender": "Men" if gender == "M" else "Women",
                "n_athlete_seasons": len(stats_df),
                "median_raw_sd_sec": round(med_raw, 2),
                "median_converted_only_sd_sec": round(med_conv, 2),
                "median_std_sd_sec": round(med_std, 2),
                "pct_reduction_std_vs_raw": round(100 * (1 - med_std / med_raw), 1) if med_raw else None,
                "pct_reduction_std_vs_converted": round(100 * (1 - med_std / med_conv), 1)
                if med_conv
                else None,
            }
        )

    all_stats = pd.DataFrame(all_season_stats) if all_season_stats else pd.DataFrame()
    summary = {"min_distinct_meets_per_athlete_season": min_races, "by_gender": rows_out}
    if len(all_stats):
        med_raw = all_stats["raw_sd"].median()
        med_conv = all_stats["conv_sd"].median()
        med_std = all_stats["std_sd"].median()
        summary["pooled_median_raw_sd"] = round(med_raw, 2)
        summary["pooled_median_converted_only_sd"] = round(med_conv, 2)
        summary["pooled_median_std_sd"] = round(med_std, 2)
        summary["pooled_pct_reduction_std_vs_raw"] = round(100 * (1 - med_std / med_raw), 1)
        summary["pooled_pct_reduction_std_vs_converted"] = round(
            100 * (1 - med_std / med_conv), 1
        )
    return summary
