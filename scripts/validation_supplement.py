"""Extended robustness checks (Riegel, track, cross-sport, finisher censoring)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import COMPREHENSIVE_FROM, RIEGEL_B_MEN, RIEGEL_B_WOMEN
from events import event_category
from load_data import build_results_frame
from standardization import compute_xc_times
from utils import (
    adjust_time_for_race,
    applies_wind_conversion,
    get_course_details,
    parse_time,
    track_venue_factor_to_outdoor_flat,
    wind_factor,
)
from validation_improvement import improvement_summary
from xc_frame import build_xc_frame, prepare_xc_results


def _variance_pct_by_gender(tables: dict, *, min_meets: int = 3, **riegel_kw) -> dict[str, float | None]:
    timed = compute_xc_times(build_xc_frame(tables, exclude_nationals=True), **riegel_kw)
    perf = timed[np.isfinite(timed["raw_sec"]) & np.isfinite(timed["standardized_sec"])].copy()
    perf = perf.dropna(subset=["season"])
    out: dict[str, float | None] = {}
    for gender, label in [("M", "Men"), ("F", "Women")]:
        sds_raw, sds_std = [], []
        g = perf[perf["gender"] == gender]
        for (_, _), grp in g.groupby(["athlete_id", "season"]):
            if grp["meet_id"].nunique() < min_meets:
                continue
            r_sd, s_sd = grp["raw_sec"].std(), grp["standardized_sec"].std()
            if np.isfinite(r_sd) and np.isfinite(s_sd) and r_sd > 0:
                sds_raw.append(r_sd)
                sds_std.append(s_sd)
        if sds_raw:
            med_r, med_s = float(np.median(sds_raw)), float(np.median(sds_std))
            out[label] = round(100.0 * (1.0 - med_s / med_r), 1)
        else:
            out[label] = None
    return out


def riegel_exponent_sensitivity(tables: dict) -> dict:
    """Compare within-athlete variance reduction under alternate Riegel exponents."""
    baseline = _variance_pct_by_gender(tables)
    configs = [
        ("gender_specific_default", {}),
        ("unified_b_1.06", {"riegel_b_unified": 1.06}),
        ("unified_b_1.055", {"riegel_b_unified": 1.055}),
        ("unified_b_1.08", {"riegel_b_unified": 1.08}),
        ("men_1.055_women_1.08_explicit", {"riegel_b_men": 1.055, "riegel_b_women": 1.08}),
    ]
    by_config = {}
    for name, kw in configs:
        pct = _variance_pct_by_gender(tables, **kw)
        delta_pp = {
            g: round(pct[g] - baseline[g], 2) if pct.get(g) is not None and baseline.get(g) is not None else None
            for g in ("Men", "Women")
        }
        by_config[name] = {
            "pct_reduction_std_vs_raw": pct,
            "delta_pp_vs_gender_specific_default": delta_pp,
        }
    max_abs_delta = max(
        abs(v)
        for cfg in by_config.values()
        for v in cfg["delta_pp_vs_gender_specific_default"].values()
        if v is not None
    )
    return {
        "default_exponents": {"men": RIEGEL_B_MEN, "women": RIEGEL_B_WOMEN},
        "by_config": by_config,
        "max_abs_delta_pp_any_gender": round(max_abs_delta, 2),
    }


def finisher_censoring_sensitivity(tables: dict, df: pd.DataFrame) -> dict:
    """Export is finisher-only; stress improvement stats by minimum races per season."""
    xc = df[df["sport_name"] == "Cross Country"].copy()
    xc["raw_sec"] = xc["result_time"].map(parse_time)
    xc_finite = int(np.isfinite(xc["raw_sec"]).sum())
    unparseable_by_sport = {}
    for sport, g in df.groupby("sport_name"):
        secs = g["result_time"].map(parse_time)
        unparseable_by_sport[sport] = int((~np.isfinite(secs)).sum())

    min_race_grid = {}
    for min_r in (2, 3, 4):
        summ = improvement_summary(tables, min_races=min_r)
        infl = summ["converted_inflation_vs_standardized"]
        min_race_grid[str(min_r)] = {
            "n_athlete_seasons": {
                "men": summ["by_mode"]["converted_only"]["men"]["n_athlete_seasons"],
                "women": summ["by_mode"]["converted_only"]["women"]["n_athlete_seasons"],
            },
            "pct_inflation_vs_full": {
                k: (v["pct_inflation_vs_full"] if v else None) for k, v in infl.items()
            },
        }

    prep = prepare_xc_results(tables, exclude_nationals=True)
    prep["raw_sec"] = prep["result_time"].map(parse_time)
    prep = prep[np.isfinite(prep["raw_sec"])]
    races_per_as = prep.groupby(["athlete_id", "season"]).size()
    return {
        "export_note": "Public export contains approved finisher times only; DNS/DNF are not rows.",
        "xc_approved_results": int(len(xc)),
        "xc_parseable_times": xc_finite,
        "unparseable_times_by_sport": unparseable_by_sport,
        "xc_races_per_athlete_season": {
            "p50": float(races_per_as.median()),
            "p90": float(races_per_as.quantile(0.9)),
            "pct_with_3plus_races": round(100.0 * (races_per_as >= 3).mean(), 1),
        },
        "improvement_inflation_by_min_races": min_race_grid,
    }


def track_wind_venue_validation(df: pd.DataFrame, tables: dict) -> dict:
    """Outdoor track: wind and venue factor coverage and magnitude (comprehensive era)."""
    cut = pd.Timestamp(COMPREHENSIVE_FROM)
    ot = df[(df["sport_name"] == "Outdoor Track") & (df["meet_date"] >= cut)].copy()
    cd = tables["course_details"]

    wind_factors = []
    track_factors = []
    combined_factors = []
    sprint_wind_recorded = 0
    sprint_wind_applicable = 0
    n_rows = 0

    for _, row in ot.iterrows():
        raw = parse_time(row["result_time"])
        if not np.isfinite(raw) or raw <= 0:
            continue
        n_rows += 1
        event = row["event_name"]
        sport = row["sport_name"]
        gender = row["gender"]
        wind = row.get("wind")

        if applies_wind_conversion(event, sport):
            sprint_wind_applicable += 1
            if wind is not None and not (isinstance(wind, float) and np.isnan(wind)):
                sprint_wind_recorded += 1
            wf = wind_factor(event, gender, wind, sport_name=sport)
            wind_factors.append(wf)

        tf = track_venue_factor_to_outdoor_flat(event, gender, sport_name=sport)
        track_factors.append(tf)

        cd_rec = get_course_details(row, cd)
        adj = adjust_time_for_race(
            event,
            row["result_time"],
            cd_rec,
            gender,
            row["altitude"],
            sport_name=sport,
            wind_mps=wind,
        )
        if np.isfinite(adj) and adj > 0:
            combined_factors.append(adj / raw)

    def _fac_summary(vals: list[float]) -> dict:
        x = np.array(vals, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return {"n": 0}
        return {
            "n": int(len(x)),
            "median": round(float(np.median(x)), 4),
            "mean": round(float(np.mean(x)), 4),
            "p05": round(float(np.percentile(x, 5)), 4),
            "p95": round(float(np.percentile(x, 95)), 4),
            "pct_outside_0p98_1p02": round(100.0 * ((x < 0.98) | (x > 1.02)).mean(), 2),
        }

    return {
        "era": "comprehensive",
        "n_results_parseable": n_rows,
        "sprint_hurdle_wind_applicable": sprint_wind_applicable,
        "sprint_hurdle_wind_recorded": sprint_wind_recorded,
        "pct_sprint_wind_recorded": round(100.0 * sprint_wind_recorded / sprint_wind_applicable, 1)
        if sprint_wind_applicable
        else None,
        "wind_factor_among_applicable": _fac_summary(wind_factors),
        "track_venue_factor_all": _fac_summary(track_factors),
        "full_pipeline_factor_raw_to_adjusted": _fac_summary(combined_factors),
        "meet_level_weather_pct_from_metadata_table": round(
            100.0 * ot.groupby("meet_id")["has_weather_meta"].any().mean(), 1
        ),
    }


def _sport_standardization_summary(
    sport_name: str,
    sub: pd.DataFrame,
    tables: dict,
) -> dict:
    cd = tables["course_details"]
    factors: list[float] = []

    if sport_name == "Cross Country":
        ids = set(sub["result_id"])
        frame = build_xc_frame(tables, exclude_nationals=False)
        frame = frame[frame["result_id"].isin(ids)]
        timed = compute_xc_times(frame)
        mask = np.isfinite(timed["raw_sec"]) & np.isfinite(timed["standardized_sec"])
        raw = timed.loc[mask, "raw_sec"].to_numpy()
        adj = timed.loc[mask, "standardized_sec"].to_numpy()
        factors.extend((adj / raw).tolist())
    else:
        for _, row in sub.iterrows():
            raw = parse_time(row["result_time"])
            if not np.isfinite(raw) or raw <= 0:
                continue
            if event_category(str(row["event_name"])) in ("field", "relay", "other"):
                continue
            cd_rec = get_course_details(row, cd)
            adj = adjust_time_for_race(
                row["event_name"],
                row["result_time"],
                cd_rec,
                row["gender"],
                row["altitude"],
                sport_name=sport_name,
                wind_mps=row.get("wind"),
            )
            if not np.isfinite(adj) or adj <= 0:
                continue
            factors.append(adj / raw)

    fac = np.array(factors, dtype=float)
    fac = fac[np.isfinite(fac)]
    med_f = float(np.median(fac)) if len(fac) else None
    med_abs_log = float(np.median(np.abs(np.log(fac)))) if len(fac) else None
    return {
        "n_results_standardized": int(len(fac)),
        "median_factor": round(med_f, 4) if med_f is not None else None,
        "median_abs_log_factor": round(med_abs_log, 4) if med_abs_log is not None else None,
        "pct_factor_outside_0p98_1p02": round(100.0 * ((fac < 0.98) | (fac > 1.02)).mean(), 2)
        if len(fac)
        else None,
    }


def cross_sport_standardization_checks(df: pd.DataFrame, tables: dict) -> dict:
    """Per-sport adjustment factor summaries (comprehensive era, distance running events)."""
    cut = pd.Timestamp(COMPREHENSIVE_FROM)
    comp = df[df["meet_date"] >= cut]
    by_sport = {}
    for sport in sorted(comp["sport_name"].unique()):
        sub = comp[comp["sport_name"] == sport]
        by_sport[sport] = _sport_standardization_summary(sport, sub, tables)
    return {"era": "comprehensive", "by_sport": by_sport}
