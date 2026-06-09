"""Aggregate validation and descriptive statistics for the NRCD export.

Writes results/supplement_stats.json. Optionally refreshes ml_standardization_r2.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from altitude import apply_meet_altitude
from config import COMPREHENSIVE_FROM, HEAT_QUADRATIC_COEFF, RESULTS, XC_TARGET_F, XC_TARGET_M
from events import parse_event_distance_m
from load_data import build_results_frame, load_tables
from standardization import (
    _elevation_factor_array,
    _weather_factor_array,
    compute_xc_times,
)
from stats_longitudinal import school_year_start
from validation_improvement import improvement_summary
from validation_supplement import (
    cross_sport_standardization_checks,
    finisher_censoring_sensitivity,
    riegel_exponent_sensitivity,
    track_wind_venue_validation,
)
from validation_variance import within_athlete_variance
from xc_frame import build_xc_frame

# Approximate 5th-finisher pace among top-15 teams at division nationals (8 km / 6 km).
# Collegiate values from published championship team results; not fitted to NRCD.
COLLEGIATE_NATIONALS_ANCHORS = {
    "Men": {
        "ncaa_d1": 26 * 60 + 0,
        "ncaa_d2": 28 * 60 + 0,
        "ncaa_d3": 29 * 60 + 30,
        "naia": 28 * 60 + 30,
        "njcaa": 28 * 60 + 30,
    },
    "Women": {
        "ncaa_d1": 22 * 60 + 0,
        "ncaa_d2": 24 * 60 + 0,
        "ncaa_d3": 25 * 60 + 30,
        "naia": 24 * 60 + 30,
        "njcaa": 25 * 60 + 30,
    },
}


def _nirca_nationals_fifth_man_anchor(
    tables: dict,
    *,
    min_year: int = 2015,
    n_teams: int = 15,
    scorer_rank: int = 5,
) -> dict[str, int]:
    """Median standardized 5th-finisher time on top-15 teams at NIRCA nationals."""
    meet = tables["meet"]
    timed = compute_xc_times(build_xc_frame(tables, exclude_nationals=False))
    nat_ids = set(meet[meet["nationals"].astype(bool)]["meet_id"])
    nat = timed[
        timed["meet_id"].isin(nat_ids)
        & timed["event_category"].eq("distance")
        & (pd.to_datetime(timed["race_date"]).dt.year >= min_year)
    ]
    out: dict[str, int] = {}
    for gender, label in [("M", "Men"), ("F", "Women")]:
        vals: list[float] = []
        for _, grp_meet in nat.loc[nat["gender"] == gender].groupby("meet_id"):
            rows = []
            for _, grp in grp_meet.groupby("team_id"):
                times = grp["standardized_sec"].dropna().sort_values().values
                if len(times) >= scorer_rank:
                    rows.append(times[scorer_rank - 1])
            if len(rows) < n_teams:
                continue
            top = np.sort(rows)[:n_teams]
            vals.extend(top.tolist())
        if not vals:
            raise ValueError(f"No NIRCA nationals fifth-man samples for {label}")
        out[label] = int(round(float(np.median(vals)) / 30) * 30)
    return out

def _build_ncaa_reference_times(nirca_fifth_man_sec: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        "Men": {
            "ncaa_d1_championship_contender": 23 * 60 + 30,
            "ncaa_d1_all_american_approx": 24 * 60 + 0,
            "ncaa_d1_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Men"]["ncaa_d1"],
            "ncaa_d2_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Men"]["ncaa_d2"],
            "ncaa_d3_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Men"]["ncaa_d3"],
            "naia_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Men"]["naia"],
            "njcaa_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Men"]["njcaa"],
            "nirca_nationals_fifth_man": nirca_fifth_man_sec["Men"],
            "recreational_5k_equivalent": 30 * 60 + 0,
        },
        "Women": {
            "ncaa_d1_championship_contender": 19 * 60 + 30,
            "ncaa_d1_all_american_approx": 20 * 60 + 0,
            "ncaa_d1_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Women"]["ncaa_d1"],
            "ncaa_d2_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Women"]["ncaa_d2"],
            "ncaa_d3_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Women"]["ncaa_d3"],
            "naia_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Women"]["naia"],
            "njcaa_nationals_fifth_man": COLLEGIATE_NATIONALS_ANCHORS["Women"]["njcaa"],
            "nirca_nationals_fifth_man": nirca_fifth_man_sec["Women"],
            "recreational_5k_equivalent": 26 * 60 + 0,
        },
    }


def _pctile(vals: np.ndarray, ps: list[float]) -> dict[str, float | None]:
    clean = vals[np.isfinite(vals)]
    if len(clean) == 0:
        return {f"p{int(p)}": None for p in ps}
    return {f"p{int(p)}": round(float(np.percentile(clean, p)), 4) for p in ps}


def _summarize_factor(arr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    x = arr if mask is None else arr[mask]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "median": round(float(np.median(x)), 4),
        "mean": round(float(np.mean(x)), 4),
        "p05": round(float(np.percentile(x, 5)), 4),
        "p95": round(float(np.percentile(x, 95)), 4),
        "pct_outside_0p95_1p05": round(100.0 * ((x < 0.95) | (x > 1.05)).mean(), 2),
    }


def _median_sd_by_clock(perf: pd.DataFrame, col: str, min_meets: int = 3) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for gender in ["M", "F"]:
        g = perf[perf["gender"] == gender]
        sds = []
        for (_, _), grp in g.groupby(["athlete_id", "season"]):
            if grp["meet_id"].nunique() < min_meets:
                continue
            v = grp[col].std()
            if np.isfinite(v):
                sds.append(v)
        label = "Men" if gender == "M" else "Women"
        med = float(np.median(sds)) if sds else None
        out[label] = {
            "n_athlete_seasons": len(sds),
            "median_sd_sec": round(med, 2) if med is not None else None,
        }
    return out


def xc_clocks_with_pre_weather(tables: dict) -> pd.DataFrame:
    timed = compute_xc_times(build_xc_frame(tables, exclude_nationals=True))
    out = timed[np.isfinite(timed["raw_sec"])].copy()
    temp = out.get("cd_temperature", pd.Series(np.nan, index=out.index)).to_numpy(dtype=float)
    dew = out.get("cd_dew_point", pd.Series(np.nan, index=out.index)).to_numpy(dtype=float)
    wf = _weather_factor_array(temp, dew, k=HEAT_QUADRATIC_COEFF)
    wf = np.where(np.isfinite(wf) & (wf > 0), wf, 1.0)
    std = out["standardized_sec"].to_numpy(dtype=float)
    out["pre_weather_std_sec"] = np.where(np.isfinite(std), std / wf, np.nan)
    return out


def variance_stepwise_ablation(tables: dict, min_meets: int = 3) -> dict:
    perf = xc_clocks_with_pre_weather(tables).dropna(subset=["season"])
    stages = [
        ("raw_sec", "raw"),
        ("converted_sec", "converted_only"),
        ("pre_weather_std_sec", "pre_weather_env"),
        ("standardized_sec", "fully_standardized"),
    ]
    by_gender: dict[str, dict] = {}
    for col, key in stages:
        for gender, row in _median_sd_by_clock(perf, col, min_meets=min_meets).items():
            by_gender.setdefault(gender, {})[key] = row
    return {"min_meets_per_athlete_season": min_meets, "by_gender": by_gender}


def xc_factor_distributions(tables: dict) -> dict:
    base = build_xc_frame(tables, exclude_nationals=True)
    base = base[base["race_date"] >= pd.Timestamp(COMPREHENSIVE_FROM)].copy()

    d_reported = base["event_name"].map(parse_event_distance_m).to_numpy(dtype=float)
    d_actual = base.get("cd_estimated_course_distance", pd.Series(np.nan, index=base.index)).to_numpy(
        dtype=float
    )
    d_actual = np.where(np.isfinite(d_actual), d_actual, d_reported)

    temp = base.get("cd_temperature", pd.Series(np.nan, index=base.index)).to_numpy(dtype=float)
    dew = base.get("cd_dew_point", pd.Series(np.nan, index=base.index)).to_numpy(dtype=float)
    gain = base.get("cd_elevation_gain", pd.Series(np.nan, index=base.index)).to_numpy(dtype=float)
    loss = base.get("cd_elevation_loss", pd.Series(np.nan, index=base.index)).to_numpy(dtype=float)

    wf = _weather_factor_array(temp, dew)
    ef = _elevation_factor_array(gain, loss, d_actual, d_reported)
    heat = temp + dew
    hot = np.isfinite(heat) & (heat > 100)

    alt_factors = np.ones(len(base), dtype=float)
    ref_t = 1000.0
    for i, row in enumerate(base.itertuples()):
        pb = getattr(row, "cd_barometric_pressure", None)
        if pb is not None and (isinstance(pb, float) and np.isnan(pb)):
            pb = None
        alt_factors[i] = apply_meet_altitude(
            ref_t,
            row.event_name,
            row.altitude,
            row.gender,
            barometric_pressure_hpa=pb,
        ) / ref_t

    elev_finite = ef[np.isfinite(ef) & (ef > 0)]
    return {
        "era": "comprehensive",
        "n_results": int(len(base)),
        "weather_factor_all": _summarize_factor(wf),
        "weather_factor_hot_H_gt_100": _summarize_factor(wf, hot),
        "elevation_factor": _summarize_factor(ef),
        "altitude_factor": _summarize_factor(alt_factors),
        "median_abs_log_elevation_factor": round(float(np.median(np.abs(np.log(elev_finite)))), 4)
        if len(elev_finite)
        else None,
    }


def metadata_coverage_meet_vs_result(df: pd.DataFrame) -> list[dict]:
    rows = []
    flags = [
        ("weather", "has_weather_meta"),
        ("course_features", "has_course_features_meta"),
        ("meet_altitude", "has_meet_altitude"),
    ]
    for (sport, era), g in df.groupby(["sport_name", "era"]):
        row = {
            "sport": sport,
            "era": era,
            "n_results": int(len(g)),
            "n_meets": int(g["meet_id"].nunique()),
        }
        for label, col in flags:
            if col not in g.columns:
                continue
            row[f"{label}_result_pct"] = round(100.0 * g[col].mean(), 1)
            row[f"{label}_meet_pct"] = round(100.0 * g.groupby("meet_id")[col].any().mean(), 1)
        rows.append(row)
    return rows


def _xc_career_pr_frame(tables: dict) -> pd.DataFrame:
    timed = compute_xc_times(build_xc_frame(tables, exclude_nationals=False))
    timed = timed[timed["event_category"] == "distance"]
    timed = timed[np.isfinite(timed["standardized_sec"])].copy()
    timed["season"] = school_year_start(timed["race_date"])
    timed["career_pr_std"] = timed.groupby(["athlete_id", "gender"])["standardized_sec"].transform("min")
    return timed


def _career_pr_cohort_summary(
    timed: pd.DataFrame,
    ncaa_reference_times: dict[str, dict[str, int]],
    *,
    min_school_years: int = 1,
    min_races_per_season: int | None = None,
) -> dict[str, dict]:
    """Career PR percentiles for athletes meeting longitudinal depth filters."""
    season_stats = (
        timed.groupby(["athlete_id", "season"])
        .agg(n_results=("result_id", "count"), n_meets=("meet_id", "nunique"))
        .reset_index()
    )
    if min_races_per_season is not None:
        season_stats = season_stats[season_stats["n_results"] >= min_races_per_season]
    n_school_years = season_stats.groupby("athlete_id")["season"].nunique()
    eligible = set(n_school_years[n_school_years >= min_school_years].index.astype(int))

    percentile_ps = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    out: dict[str, dict] = {}
    for gender, label in [("M", "Men"), ("F", "Women")]:
        pr = (
            timed.loc[(timed["gender"] == gender) & (timed["athlete_id"].isin(eligible))]
            .drop_duplicates(["athlete_id"])["career_pr_std"]
            .values
        )
        ref_pct = {
            name: round(float(100.0 * (pr <= sec).mean()), 1) if len(pr) else None
            for name, sec in ncaa_reference_times[label].items()
        }
        out[label] = {
            "n_athletes": int(len(pr)),
            "career_pr_standardized_sec_percentiles": _pctile(pr, percentile_ps),
            "pct_athletes_at_or_faster_than_anchor": ref_pct,
        }
    return out


def nirca_collegiate_overlap(
    tables: dict,
    *,
    nirca_fifth_man_sec: dict[str, int],
    collegiate_anchors: dict[str, dict[str, int]],
) -> dict:
    """Overlap of career standardized PRs with NIRCA vs.\ collegiate nationals depth anchors."""
    timed = _xc_career_pr_frame(tables)
    out: dict = {
        "unit": "career_pr_best_standardized_sec",
        "anchor_definition": "5th finisher among top-15 teams at nationals (standardized sec)",
        "nirca_nationals_fifth_man_sec": nirca_fifth_man_sec,
        "collegiate_nationals_anchors_sec": collegiate_anchors,
        "by_gender": {},
    }
    for gender, label in [("M", "Men"), ("F", "Women")]:
        pr = (
            timed.loc[timed["gender"] == gender]
            .drop_duplicates(["athlete_id"])["career_pr_std"]
            .values
        )
        pr = pr[np.isfinite(pr)]
        nirca_sec = nirca_fifth_man_sec[label]
        nirca_mask = pr <= nirca_sec
        n_nirca = int(nirca_mask.sum())
        gender_out = {
            "n_athletes": int(len(pr)),
            "pct_at_or_faster_than_nirca_fifth_man": round(100.0 * nirca_mask.mean(), 1) if len(pr) else None,
            "by_collegiate_division": {},
        }
        for div, anchor_sec in collegiate_anchors[label].items():
            div_mask = pr <= anchor_sec
            both_mask = nirca_mask & div_mask
            n_div = int(div_mask.sum())
            n_both = int(both_mask.sum())
            gender_out["by_collegiate_division"][div] = {
                "anchor_sec": anchor_sec,
                "pct_all_at_or_faster_anchor": round(100.0 * div_mask.mean(), 1) if len(pr) else None,
                "pct_all_meeting_both": round(100.0 * both_mask.mean(), 1) if len(pr) else None,
                "pct_nirca_fifth_man_only_not_div": round(100.0 * (nirca_mask & ~div_mask).mean(), 1)
                if len(pr)
                else None,
                "pct_div_only_not_nirca": round(100.0 * (div_mask & ~nirca_mask).mean(), 1)
                if len(pr)
                else None,
                "pct_of_nirca_fifth_man_also_at_div_pace": round(100.0 * (pr[nirca_mask] <= anchor_sec).mean(), 1)
                if n_nirca
                else None,
                "pct_of_div_also_at_nirca_fifth_man_pace": round(100.0 * (pr[div_mask] <= nirca_sec).mean(), 1)
                if n_div
                else None,
                "n_at_or_faster_anchor": n_div,
                "n_meeting_both": n_both,
                "n_nirca_fifth_man_only": int((nirca_mask & ~div_mask).sum()),
                "n_div_only": int((div_mask & ~nirca_mask).sum()),
            }
        out["by_gender"][label] = gender_out
    return out


def xc_time_percentiles(tables: dict, *, ncaa_reference_times: dict[str, dict[str, int]]) -> dict:
    """Career PR cohorts: all athletes vs.\ longitudinal subsets (school-year depth)."""
    timed = _xc_career_pr_frame(tables)
    percentile_ps = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    cohorts = {
        "all_athletes": _career_pr_cohort_summary(timed, ncaa_reference_times, min_school_years=1),
        "min_2_school_years": _career_pr_cohort_summary(timed, ncaa_reference_times, min_school_years=2),
        "min_3_school_years": _career_pr_cohort_summary(timed, ncaa_reference_times, min_school_years=3),
        "min_2_school_years_min_2_races_per_season": _career_pr_cohort_summary(
            timed, ncaa_reference_times, min_school_years=2, min_races_per_season=2
        ),
    }

    all_m = cohorts["all_athletes"]["Men"]
    all_f = cohorts["all_athletes"]["Women"]
    return {
        "unit": "career_pr_best_standardized_sec",
        "percentile_points": percentile_ps,
        "cohorts": cohorts,
        # Legacy flat layout (all athletes)
        "Men": {
            "n_athletes": all_m["n_athletes"],
            "distance_km": XC_TARGET_M / 1000.0,
            **all_m,
            "n_athlete_seasons": all_m["n_athletes"],
            "best_standardized_sec_percentiles": all_m["career_pr_standardized_sec_percentiles"],
            "pct_athlete_seasons_at_or_faster_than_anchor": all_m["pct_athletes_at_or_faster_than_anchor"],
        },
        "Women": {
            "n_athletes": all_f["n_athletes"],
            "distance_km": XC_TARGET_F / 1000.0,
            **all_f,
            "n_athlete_seasons": all_f["n_athletes"],
            "best_standardized_sec_percentiles": all_f["career_pr_standardized_sec_percentiles"],
            "pct_athlete_seasons_at_or_faster_than_anchor": all_f["pct_athletes_at_or_faster_than_anchor"],
        },
    }


def longitudinal_by_sport(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["school_year"] = school_year_start(df["meet_date"])
    out = {}
    for sport in sorted(df["sport_name"].unique()):
        s = df[df["sport_name"] == sport]
        per_asy = (
            s.groupby(["athlete_id", "school_year"])
            .agg(n_results=("result_id", "count"), n_meets=("meet_id", "nunique"))
            .reset_index()
        )
        athletes = s["athlete_id"].nunique()
        seasons = per_asy.groupby("athlete_id")["school_year"].nunique()
        out[sport] = {
            "n_athletes": int(athletes),
            "n_athlete_seasons": int(len(per_asy)),
            "pct_athletes_2plus_seasons": round(100.0 * (seasons >= 2).sum() / athletes, 1) if athletes else None,
            "results_per_athlete_season": _pctile(per_asy["n_results"].values, [25, 50, 75, 90, 95]),
            "meets_per_athlete_season": _pctile(per_asy["n_meets"].values, [25, 50, 75, 90, 95]),
        }
    return out


def meet_size_skew(df: pd.DataFrame) -> dict:
    cut = pd.Timestamp(COMPREHENSIVE_FROM)
    xc = df[(df["sport_name"] == "Cross Country") & (df["meet_date"] >= cut)]
    sizes = xc.groupby("meet_id").size().sort_values(ascending=False)
    total = int(sizes.sum())
    return {
        "comprehensive_xc_meets": int(sizes.shape[0]),
        "comprehensive_xc_results": total,
        "pct_results_in_top_10_meets": round(100.0 * sizes.head(10).sum() / total, 1) if total else None,
        "pct_results_in_top_25_meets": round(100.0 * sizes.head(25).sum() / total, 1) if total else None,
        "median_results_per_meet": int(sizes.median()) if len(sizes) else None,
        "max_results_per_meet": int(sizes.max()) if len(sizes) else None,
    }


def xc_gender_over_time(df: pd.DataFrame) -> list[dict]:
    xc = df[df["sport_name"] == "Cross Country"].copy()
    xc["school_year"] = school_year_start(xc["meet_date"])
    rows = []
    for yr, g in xc.groupby("school_year"):
        if pd.isna(yr):
            continue
        rows.append(
            {
                "school_year": int(yr),
                "n_results": int(len(g)),
                "women_pct": round(100.0 * (g["gender"] == "F").mean(), 1),
            }
        )
    return sorted(rows, key=lambda r: r["school_year"])


def meets_missing_course_features(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    cut = pd.Timestamp(COMPREHENSIVE_FROM)
    xc = df[(df["sport_name"] == "Cross Country") & (df["meet_date"] >= cut)]
    by_meet = (
        xc.groupby("meet_id")
        .agg(
            n_results=("result_id", "count"),
            course_pct=("has_course_features_meta", "mean"),
            weather_pct=("has_weather_meta", "mean"),
        )
        .reset_index()
    )
    missing = by_meet[by_meet["course_pct"] < 1.0].sort_values("n_results", ascending=False).head(top_n)
    return [
        {
            "meet_id": int(r.meet_id),
            "n_results": int(r.n_results),
            "course_features_pct": round(100.0 * r.course_pct, 1),
            "weather_pct": round(100.0 * r.weather_pct, 1),
        }
        for r in missing.itertuples()
    ]


def main(*, run_ml: bool = True, progress_disable: bool = False) -> None:
    tables = load_tables()
    df = build_results_frame(tables)
    imp = improvement_summary(tables)
    nirca_fifth_man_sec = _nirca_nationals_fifth_man_anchor(tables)
    ncaa_reference_times = _build_ncaa_reference_times(nirca_fifth_man_sec)

    race_frames = None
    try:
        from validation_ml_r2 import _xc_race_frames, meet_holdout_summary

        race_frames = _xc_race_frames(tables, progress_disable=progress_disable)
        ml_meet_holdout = meet_holdout_summary(race_frames)
    except ImportError:
        ml_meet_holdout = None

    payload = {
        "variance_stepwise_ablation": variance_stepwise_ablation(tables),
        "xc_factor_distributions": xc_factor_distributions(tables),
        "variance_validation": within_athlete_variance(tables),
        "improvement_summary": {
            "pct_positive_improvement": imp["pct_positive_improvement"],
            "converted_inflation_vs_standardized": imp["converted_inflation_vs_standardized"],
        },
        "metadata_coverage_meet_vs_result": metadata_coverage_meet_vs_result(df),
        "xc_time_percentiles": xc_time_percentiles(tables, ncaa_reference_times=ncaa_reference_times),
        "nirca_collegiate_overlap": nirca_collegiate_overlap(
            tables,
            nirca_fifth_man_sec=nirca_fifth_man_sec,
            collegiate_anchors=COLLEGIATE_NATIONALS_ANCHORS,
        ),
        "longitudinal_by_sport": longitudinal_by_sport(df),
        "meet_size_skew": meet_size_skew(df),
        "xc_gender_over_time": xc_gender_over_time(df),
        "meets_missing_course_features_top10": meets_missing_course_features(df),
        "ncaa_reference_times_sec": ncaa_reference_times,
        "nirca_nationals_fifth_man_sec": nirca_fifth_man_sec,
        "collegiate_nationals_anchors_sec": COLLEGIATE_NATIONALS_ANCHORS,
        "riegel_exponent_sensitivity": riegel_exponent_sensitivity(tables),
        "finisher_censoring_sensitivity": finisher_censoring_sensitivity(tables, df),
        "track_wind_venue_validation": track_wind_venue_validation(df, tables),
        "cross_sport_standardization_checks": cross_sport_standardization_checks(df, tables),
        "ml_meet_holdout_temporal": ml_meet_holdout,
    }

    out_path = RESULTS / "supplement_stats.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_path}")

    if run_ml:
        from validation_ml_r2 import main as ml_r2_main

        ml_r2_main(progress_disable=progress_disable)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-ml", action="store_true", help="Skip ML R² bootstrap (~2 min).")
    p.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    a = p.parse_args()
    main(run_ml=not a.no_ml, progress_disable=a.quiet)
