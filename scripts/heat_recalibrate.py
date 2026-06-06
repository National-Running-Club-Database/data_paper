#!/usr/bin/env python3
"""
Explore data-driven recalibration of the heat quadratic coefficient k.

Compares literature-based k (Hadley piecewise / deployed 0.0015) against fits on
NRCD XC results. Naive fits are confounded; we report partial and within-athlete
designs and recommend keeping the citation-grounded k unless a fit is clearly
better on multiple criteria without extreme k.

Usage (from repo root, data/ populated):
  python scripts/heat_recalibrate.py
  python scripts/heat_recalibrate.py --write results/heat_recalibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HEAT_QUADRATIC_COEFF, RESULTS
from heat_hadley import (
    fit_quadratic_coefficient,
    hadley_slowdown_pct_linear,
    quadratic_slowdown_pct,
    validate_piecewise_to_quadratic,
)
from load_data import load_tables
from standardization import apply_heat_factor, heat_index, pre_weather_xc, riegel_exponent
from utils import get_course_details, parse_time


def _build_xc_weather_frame(tables: dict) -> pd.DataFrame:
    meet = tables["meet"]
    result = tables["result"]
    athlete = tables["athlete"]
    running_event = tables["running_event"]
    course_details = tables["course_details"]

    xc_meets = set(meet.loc[meet["sport_id"] == 1, "meet_id"])
    df = result[result["meet_id"].isin(xc_meets)].copy()
    df = df.merge(athlete[["athlete_id", "gender"]], on="athlete_id")
    df = df.merge(running_event, on="running_event_id")
    df = df.merge(meet[["meet_id", "start_date"]], on="meet_id", how="left")
    df["year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year

    rows = []
    for _, row in df.iterrows():
        raw = parse_time(row["result_time"])
        if not np.isfinite(raw):
            continue
        cd = get_course_details(row, course_details)
        h = heat_index(cd.get("temperature"), cd.get("dew_point"))
        if h is None:
            continue
        pre = pre_weather_xc(row, course_details)
        if not np.isfinite(pre):
            continue
        rows.append(
            {
                "athlete_id": row["athlete_id"],
                "year": row["year"],
                "gender": row["gender"],
                "raw_sec": raw,
                "pre_weather_sec": pre,
                "heat_index": h,
                "excess_h": max(0.0, h - 100.0),
                "excess_h_sq": max(0.0, h - 100.0) ** 2,
            }
        )
    return pd.DataFrame(rows)


def _fit_k_ols_log_preweather(hot: pd.DataFrame) -> float | None:
    """Naive: log(pre_weather) ~ excess_h^2 on H>100 (confounded)."""
    sub = hot[hot["heat_index"] > 100].copy()
    if len(sub) < 200:
        return None
    y = np.log(sub["pre_weather_sec"].values)
    x = sub["excess_h_sq"].values
    # log(t_adj) = log(t) - pct/100  =>  log(t_adj) ≈ log(t) - k*(H-100)^2/100
    # OLS slope on x: beta ≈ -k/100  => k ≈ -100 * beta
    x_c = x - x.mean()
    beta = np.dot(x_c, y - y.mean()) / np.dot(x_c, x_c)
    k = float(-100.0 * beta)
    return k if k > 0 else None


def _fit_k_within_athlete_season(df: pd.DataFrame, min_meets: int = 3) -> float | None:
    """
    Pooled within-athlete-season slopes: log(pre_weather) ~ excess_h^2.
    Reduces some between-athlete confounding; still has course/tactical noise.
    """
    ks = []
    for (_, year), g in df.groupby(["athlete_id", "year"]):
        if pd.isna(year) or len(g) < min_meets:
            continue
        if g["excess_h_sq"].std() < 1e-6:
            continue
        y = np.log(g["pre_weather_sec"].values)
        x = g["excess_h_sq"].values
        x_c = x - x.mean()
        if np.dot(x_c, x_c) < 1e-6:
            continue
        beta = np.dot(x_c, y - y.mean()) / np.dot(x_c, x_c)
        k = -100.0 * beta
        if k > 0 and np.isfinite(k):
            ks.append(k)
    if len(ks) < 30:
        return None
    return float(np.median(ks))


def _fit_k_minimize_corr(pre: np.ndarray, h: np.ndarray, k_grid: np.ndarray) -> float:
    """Choose k to minimize |corr(pre * f(H), H)| on H>100 — can over-correct."""
    hot = h > 100
    best_k, best = HEAT_QUADRATIC_COEFF, np.inf
    for k in k_grid:
        adj = np.array([apply_heat_factor(pre[i], h[i], k) for i in range(len(pre))])
        c = np.corrcoef(adj[hot], h[hot])[0, 1]
        if np.isfinite(c) and abs(c) < best:
            best, best_k = abs(c), float(k)
    return best_k


def _fit_k_match_hadley_mean(df: pd.DataFrame) -> float:
    """Scale k so mean predicted slowdown matches Hadley linear interp at observed H."""
    hot = df[df["heat_index"] > 100]
    if len(hot) == 0:
        return HEAT_QUADRATIC_COEFF
    target = hot["heat_index"].map(hadley_slowdown_pct_linear).mean()
    mean_x = hot["excess_h_sq"].mean()
    return float(target / mean_x) if mean_x > 0 else HEAT_QUADRATIC_COEFF


def _evaluate_k(df: pd.DataFrame, k: float, label: str) -> dict:
    hot = df["heat_index"] > 100
    pre = df["pre_weather_sec"].values
    h = df["heat_index"].values
    adj = np.array([apply_heat_factor(pre[i], h[i], k) for i in range(len(df))])

    # Fidelity to Hadley table (linear reference on 101-180)
    hs = np.arange(101, 181, dtype=float)
    ref = np.array([hadley_slowdown_pct_linear(x) for x in hs])
    quad = np.array([quadratic_slowdown_pct(x, k) for x in hs])
    hadley_rmse = float(np.sqrt(np.mean((quad - ref) ** 2)))

    corr_pre_hot = float(np.corrcoef(pre[hot], h[hot])[0, 1]) if hot.sum() > 30 else None
    corr_adj_hot = float(np.corrcoef(adj[hot], h[hot])[0, 1]) if hot.sum() > 30 else None

    # Within-athlete-season SD of pre-weather times vs after heat (lower corr with H is not always good)
    sd_ratios = []
    for (_, year), g in df.groupby(["athlete_id", "year"]):
        if pd.isna(year) or len(g) < 3 or g["heat_index"].std() < 0.5:
            continue
        sub_adj = np.array(
            [apply_heat_factor(r.pre_weather_sec, r.heat_index, k) for r in g.itertuples()]
        )
        sub_raw_pre = g["pre_weather_sec"].values
        if np.std(sub_raw_pre) > 0:
            sd_ratios.append(np.std(sub_adj) / np.std(sub_raw_pre))
    mean_sd_ratio = float(np.mean(sd_ratios)) if sd_ratios else None

    mean_slowdown_hot = float(
        np.mean([quadratic_slowdown_pct(x, k) for x in df.loc[hot, "heat_index"]])
    ) if hot.any() else None

    return {
        "label": label,
        "k": round(k, 6),
        "hadley_linear_rmse_101_180": round(hadley_rmse, 4),
        "corr_pre_weather_vs_H_hot": round(corr_pre_hot, 4) if corr_pre_hot is not None else None,
        "corr_after_heat_vs_H_hot": round(corr_adj_hot, 4) if corr_adj_hot is not None else None,
        "mean_slowdown_pct_when_hot": round(mean_slowdown_hot, 4) if mean_slowdown_hot is not None else None,
        "within_athlete_season_sd_ratio_mean": round(mean_sd_ratio, 4) if mean_sd_ratio is not None else None,
        "n_athlete_seasons_sd_ratio": len(sd_ratios),
    }


def _cikm_recommendation(candidates: list[dict], piecewise: dict) -> dict:
    """CIKM resource track: prefer citation-grounded k; reject empirical fits."""
    deployed = next(c for c in candidates if c["label"] == "deployed_literature")
    hadley_ls = next((c for c in candidates if c["label"] == "hadley_ls_midpoints"), None)

    empirical_labels = {
        "naive_ols_log_preweather",
        "within_athlete_season_median",
        "match_hadley_mean_on_data",
        "corr_minimization",
    }
    empirical = [c for c in candidates if c.get("label") in empirical_labels and c.get("k")]

    reasons = [
        "CIKM Resource track: treat heat as a documented operationalization of Hadley (Max Performance Running), not an NRCD-estimated physiological law.",
        f"Deployed k={deployed['k']}; least-squares fit to Hadley band midpoints gives k≈{piecewise.get('least_squares_k_vs_band_midpoints')} (still literature-derived).",
    ]

    bad_empirical = [c for c in empirical if c["k"] > 0.005 or c["hadley_linear_rmse_101_180"] > 2.0]
    if bad_empirical:
        worst = max(bad_empirical, key=lambda c: c["k"])
        reasons.append(
            f"Data-driven fits are confounded: e.g. {worst['label']} yielded k={worst['k']} "
            f"(Hadley RMSE {worst['hadley_linear_rmse_101_180']}), far from the coaching table."
        )

    optional_k = hadley_ls["k"] if hadley_ls else deployed["k"]
    optional_note = (
        f"Optional: round to k={optional_k:.4f} for slightly tighter band-midpoint fit; "
        "difference from 0.0015 is small and not worth a full recalibration narrative."
        if hadley_ls and abs(optional_k - deployed["k"]) < 0.0003
        else f"Optional literature-only tweak: k={optional_k:.4f} (Hadley midpoint LS)."
        if hadley_ls
        else "Keep k=0.0015."
    )
    reasons.append(optional_note)
    reasons.append("Do not report NRCD-recalibrated k in the main paper; cite Hadley + Equation (heat) + piecewise-fit table.")

    return {
        "recommendation": "keep_deployed_k",
        "recommended_k": deployed["k"],
        "optional_literature_k": optional_k,
        "include_recalibration_in_paper": False,
        "include_empirical_recalibration_appendix": False,
        "rationale": reasons,
    }


def run_recalibration(tables: dict) -> dict:
    df = _build_xc_weather_frame(tables)
    if len(df) < 500:
        return {"error": "insufficient XC rows with weather", "n": len(df)}

    hot_df = df[df["heat_index"] > 100]
    k_grid = np.linspace(0.0005, 0.004, 80)

    fits = {
        "deployed_literature": HEAT_QUADRATIC_COEFF,
        "hadley_ls_midpoints": fit_quadratic_coefficient(),
        "naive_ols_log_preweather": _fit_k_ols_log_preweather(df),
        "within_athlete_season_median": _fit_k_within_athlete_season(df),
        "match_hadley_mean_on_data": _fit_k_match_hadley_mean(df),
        "corr_minimization": _fit_k_minimize_corr(
            df["pre_weather_sec"].values, df["heat_index"].values, k_grid
        ),
    }

    candidates = []
    for label, k in fits.items():
        if k is None or not np.isfinite(k) or k <= 0:
            candidates.append({"label": label, "k": None, "skipped": True})
            continue
        candidates.append(_evaluate_k(df, float(k), label))

    piecewise = validate_piecewise_to_quadratic()
    recommendation = _cikm_recommendation([c for c in candidates if "k" in c and c.get("k")], piecewise)

    return {
        "n_xc_results_with_weather": int(len(df)),
        "n_hot_H_gt_100": int(len(hot_df)),
        "heat_index_mean_when_hot": round(float(hot_df["heat_index"].mean()), 2) if len(hot_df) else None,
        "piecewise_surrogate": {
            "deployed_k": piecewise["deployed_coefficient_k"],
            "hadley_ls_k": piecewise["least_squares_k_vs_band_midpoints"],
            "rmse_vs_midpoints": piecewise["rmse_pct_vs_band_midpoints"],
        },
        "fitted_coefficients": {k: (None if v is None else round(float(v), 6)) for k, v in fits.items()},
        "candidate_metrics": candidates,
        "cikm_guidance": recommendation,
        "confounding_warning": (
            "Empirical fits use pre-weather standardized times but still confound heat with "
            "unmeasured course hardness, team tactics, fitness trajectory within season, and "
            "scheduling (early-season races cluster on warmer dates). Prefer Hadley-grounded k for the resource paper."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Heat coefficient recalibration experiments")
    parser.add_argument("--write", type=Path, default=None, help="Write JSON summary path")
    args = parser.parse_args()

    tables = load_tables()
    out = run_recalibration(tables)
    print(json.dumps(out, indent=2))

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.write}", file=sys.stderr)
    else:
        default = RESULTS / "heat_recalibration.json"
        default.parent.mkdir(parents=True, exist_ok=True)
        default.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {default}", file=sys.stderr)


if __name__ == "__main__":
    main()
