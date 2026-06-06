"""Diagnose gender asymmetry in within-athlete XC cross-meet SD reduction (raw → full std).

Hypotheses explored:
  (1) Higher raw SD baseline for women → more compressible dispersion.
  (2) Greater environmental / course-length heterogeneity before adjustment.
  (3) Distance conversion removes more variance for women than men.
  (4) Weather/elevation steps re-expand SD similarly for both after conversion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS
from load_data import load_tables
from standardization import (
    converted_only_xc,
    elevation_factor,
    heat_index,
    standardize_xc_row,
    weather_factor,
)
from utils import get_course_details, get_event_dist, parse_time


def _xc_performance_frame(tables: dict, min_meets: int = 3) -> pd.DataFrame:
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

    rows = []
    for _, row in df.iterrows():
        raw = parse_time(row["result_time"])
        if not np.isfinite(raw):
            continue
        cd = get_course_details(row, course_details)
        _, std = standardize_xc_row(row, course_details)
        conv = converted_only_xc(row, course_details)
        if not np.isfinite(std):
            continue
        d_rep = get_event_dist(row.get("event_name"))
        d_act = cd.get("estimated_course_distance")
        if d_act is None or (isinstance(d_act, float) and np.isnan(d_act)):
            d_act = d_rep
        length_ratio = (
            float(d_act) / float(d_rep) if d_rep and d_act and float(d_rep) > 0 else np.nan
        )
        h = heat_index(cd.get("temperature"), cd.get("dew_point"))
        wf = weather_factor(cd.get("temperature"), cd.get("dew_point"))
        gain, loss = cd.get("elevation_gain"), cd.get("elevation_loss")
        ef = elevation_factor(gain, loss)
        has_weather = h is not None and np.isfinite(h)

        rows.append(
            {
                "athlete_id": row["athlete_id"],
                "season": row["season"],
                "meet_id": row["meet_id"],
                "gender": row["gender"],
                "raw": raw,
                "converted_only": conv,
                "standardized": std,
                "heat_index": h,
                "weather_factor": wf,
                "elevation_factor": ef,
                "length_ratio": length_ratio,
                "has_weather": has_weather,
            }
        )

    perf = pd.DataFrame(rows).dropna(subset=["season"])
    # athlete-seasons with enough meets
    counts = perf.groupby(["athlete_id", "season", "gender"])["meet_id"].nunique()
    valid = counts[counts >= min_meets].index
    perf = perf.set_index(["athlete_id", "season", "gender"])
    perf = perf.loc[perf.index.isin(valid)].reset_index()
    return perf


def _median_sd(series: pd.Series) -> float:
    return float(series.median())


def variance_asymmetry_summary(tables: dict, min_meets: int = 3) -> dict:
    perf = _xc_performance_frame(tables, min_meets=min_meets)
    out: dict = {"min_meets_per_athlete_season": min_meets, "by_gender": {}}

    for gender, label in [("F", "Women"), ("M", "Men")]:
        g = perf[perf["gender"] == gender]
        season_stats = []
        env_stats = []
        for (_, _, _), grp in g.groupby(["athlete_id", "season", "gender"]):
            if grp["meet_id"].nunique() < min_meets:
                continue
            season_stats.append(
                {
                    "raw_sd": grp["raw"].std(),
                    "conv_sd": grp["converted_only"].std(),
                    "std_sd": grp["standardized"].std(),
                }
            )
            env_stats.append(
                {
                    "sd_heat_index": grp.loc[grp["has_weather"], "heat_index"].std()
                    if grp["has_weather"].sum() >= 2
                    else np.nan,
                    "sd_weather_factor": grp["weather_factor"].std(),
                    "sd_elevation_factor": grp["elevation_factor"].std(),
                    "sd_length_ratio": grp["length_ratio"].std(),
                    "pct_meets_with_weather": 100.0 * grp["has_weather"].mean(),
                    "mean_abs_log_weather": np.nanmean(
                        np.abs(np.log(grp["weather_factor"].clip(lower=1e-6)))
                    ),
                    "mean_abs_log_elevation": np.nanmean(
                        np.abs(np.log(grp["elevation_factor"].clip(lower=1e-6)))
                    ),
                }
            )

        ss = pd.DataFrame(season_stats)
        es = pd.DataFrame(env_stats)
        med_raw = _median_sd(ss["raw_sd"])
        med_conv = _median_sd(ss["conv_sd"])
        med_std = _median_sd(ss["std_sd"])
        pct_raw_to_conv = round(100 * (1 - med_conv / med_raw), 1) if med_raw else None
        pct_raw_to_std = round(100 * (1 - med_std / med_raw), 1) if med_raw else None
        pct_conv_to_std = round(100 * (1 - med_std / med_conv), 1) if med_conv else None

        out["by_gender"][label] = {
            "n_athlete_seasons": int(len(ss)),
            "median_raw_sd_sec": round(med_raw, 2),
            "median_converted_only_sd_sec": round(med_conv, 2),
            "median_full_std_sd_sec": round(med_std, 2),
            "pct_reduction_raw_to_converted_only": pct_raw_to_conv,
            "pct_reduction_raw_to_full_standardized": pct_raw_to_std,
            "pct_change_converted_to_full": pct_conv_to_std,
            "median_sd_heat_index_across_meets": round(float(es["sd_heat_index"].median()), 2)
            if es["sd_heat_index"].notna().any()
            else None,
            "median_sd_length_ratio_across_meets": round(
                float(es["sd_length_ratio"].median()), 4
            ),
            "median_pct_meets_with_weather": round(
                float(es["pct_meets_with_weather"].median()), 1
            ),
            "median_mean_abs_log_weather_factor": round(
                float(es["mean_abs_log_weather"].median()), 4
            ),
            "median_mean_abs_log_elevation_factor": round(
                float(es["mean_abs_log_elevation"].median()), 4
            ),
        }

    w = out["by_gender"]["Women"]["pct_reduction_raw_to_full_standardized"]
    m = out["by_gender"]["Men"]["pct_reduction_raw_to_full_standardized"]
    if w and m and m > 0:
        out["ratio_women_to_men_pct_reduction"] = round(w / m, 2)

    return out


def main() -> None:
    tables = load_tables()
    summary = variance_asymmetry_summary(tables)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "variance_asymmetry.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Variance reduction asymmetry (XC, >=3 meets/season)\n")
    for label, row in summary["by_gender"].items():
        print(f"=== {label} (n={row['n_athlete_seasons']}) ===")
        print(f"  Median cross-meet SD: raw {row['median_raw_sd_sec']}s → "
              f"converted {row['median_converted_only_sd_sec']}s → "
              f"full std {row['median_full_std_sd_sec']}s")
        print(f"  Reduction raw→converted: {row['pct_reduction_raw_to_converted_only']}%")
        print(f"  Reduction raw→full:      {row['pct_reduction_raw_to_full_standardized']}%")
        print(f"  Change converted→full:   {row['pct_change_converted_to_full']}% "
              "(negative = SD rises after weather/elev)")
        print(f"  Median SD(length ratio) across meets: {row['median_sd_length_ratio_across_meets']}")
        print(f"  Median % meets with weather: {row['median_pct_meets_with_weather']}%")
        print()

    if "ratio_women_to_men_pct_reduction" in summary:
        print(f"Women/Men ratio of full vs raw % reduction: "
              f"{summary['ratio_women_to_men_pct_reduction']}×")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
