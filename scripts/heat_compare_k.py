#!/usr/bin/env python3
"""
Compare candidate heat quadratic coefficients k (literature grid).

Validates each k against:
  - Hadley piecewise bands (midpoint RMSE, linear-interp RMSE, % of H in [101,180] inside band)
  - NRCD XC empirical plausibility (pre-weather times, H>100 subset)

Default grid includes 0.0015, 0.0016, Hadley LS (~0.00164), and lower/higher neighbors.

Usage:
  python scripts/heat_compare_k.py
  python scripts/heat_compare_k.py --k 0.0014 0.0015 0.0016 0.0018
  python scripts/heat_compare_k.py --markdown   # print GitHub-flavored table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HEAT_QUADRATIC_COEFF, RESULTS
from heat_hadley import (
    HADLEY_BANDS,
    fit_quadratic_coefficient,
    hadley_band,
    hadley_slowdown_pct_linear,
    quadratic_slowdown_pct,
)
from heat_recalibrate import _build_xc_weather_frame, _evaluate_k
from load_data import load_tables


DEFAULT_K_VALUES = [
    0.0012,
    0.0014,
    0.0015,
    0.0016,
    0.001642,  # Hadley band-midpoint least squares
    0.0018,
    0.0020,
]


def hadley_metrics_for_k(k: float) -> dict:
    """Literature fidelity metrics for a single k."""
    errors_mid = []
    for h_min, h_max, lo, hi in HADLEY_BANDS:
        if h_max <= 100:
            continue
        h_mid = (h_min + h_max) / 2.0
        quad = quadratic_slowdown_pct(h_mid, k)
        errors_mid.append(quad - (lo + hi) / 2.0)

    rmse_mid = float(np.sqrt(np.mean(np.array(errors_mid) ** 2)))

    hs = np.arange(101, 181, dtype=float)
    ref_lin = np.array([hadley_slowdown_pct_linear(h) for h in hs])
    quad_lin = np.array([quadratic_slowdown_pct(h, k) for h in hs])
    rmse_linear = float(np.sqrt(np.mean((quad_lin - ref_lin) ** 2)))

    in_band = 0
    max_below = 0.0
    max_above = 0.0
    for h in range(101, 181):
        band = hadley_band(float(h))
        if not band:
            continue
        _, _, lo, hi = band
        q = quadratic_slowdown_pct(float(h), k)
        if lo <= q <= hi:
            in_band += 1
        if q < lo:
            max_below = max(max_below, lo - q)
        if q > hi:
            max_above = max(max_above, q - hi)

    n = 80
    return {
        "rmse_vs_band_midpoints_pp": round(rmse_mid, 4),
        "rmse_vs_hadley_linear_101_180_pp": round(rmse_linear, 4),
        "pct_integer_H_inside_hadley_band": round(100 * in_band / n, 1),
        "max_pp_below_band_lo": round(max_below, 4),
        "max_pp_above_band_hi": round(max_above, 4),
        "slowdown_pct_at_H_110": round(quadratic_slowdown_pct(110, k), 4),
        "slowdown_pct_at_H_145": round(quadratic_slowdown_pct(145, k), 4),
        "slowdown_pct_at_H_160": round(quadratic_slowdown_pct(160, k), 4),
        "weather_factor_at_H_145": round(1.0 - quadratic_slowdown_pct(145, k) / 100.0, 6),
    }


def compare_k_values(
    df,
    k_values: list[float],
    *,
    deployed_k: float = HEAT_QUADRATIC_COEFF,
) -> dict:
    k_ls = fit_quadratic_coefficient()
    rows = []
    for k in sorted(set(k_values)):
        lit = hadley_metrics_for_k(k)
        emp = _evaluate_k(df, k, label=f"k={k}")
        rows.append(
            {
                "k": k,
                "is_deployed": abs(k - deployed_k) < 1e-12,
                "is_hadley_ls": abs(k - k_ls) < 1e-6,
                "literature": lit,
                "empirical": {
                    "corr_after_heat_vs_H_hot": emp.get("corr_after_heat_vs_H_hot"),
                    "corr_pre_weather_vs_H_hot": emp.get("corr_pre_weather_vs_H_hot"),
                    "mean_slowdown_pct_when_hot": emp.get("mean_slowdown_pct_when_hot"),
                    "within_athlete_season_sd_ratio_mean": emp.get(
                        "within_athlete_season_sd_ratio_mean"
                    ),
                },
            }
        )

    # Rank: primary = Hadley linear RMSE; tie-break = midpoint RMSE; prefer deployed on ties
    ranked = sorted(
        rows,
        key=lambda r: (
            r["literature"]["rmse_vs_hadley_linear_101_180_pp"],
            r["literature"]["rmse_vs_band_midpoints_pp"],
            0 if r["is_deployed"] else 1,
        ),
    )
    best = ranked[0]
    deployed_row = next(r for r in rows if r["is_deployed"])
    hadley_ls_row = next((r for r in rows if r["is_hadley_ls"]), None)
    k16_row = next((r for r in rows if abs(r["k"] - 0.0016) < 1e-12), None)

    vs_deployed = None
    if hadley_ls_row and not hadley_ls_row["is_deployed"]:
        d_lin = (
            hadley_ls_row["literature"]["rmse_vs_hadley_linear_101_180_pp"]
            - deployed_row["literature"]["rmse_vs_hadley_linear_101_180_pp"]
        )
        vs_deployed = {
            "hadley_ls_k": hadley_ls_row["k"],
            "linear_rmse_improvement_pp": round(-d_lin, 4),
            "worth_switching_from_0.0015": bool(d_lin < -0.03),
            "note": (
                "Hadley LS k only marginally closer to table; 0.0016 is a reasonable round choice."
                if abs(d_lin) < 0.05
                else "Hadley LS k measurably closer to piecewise linear reference."
            ),
        }

    return {
        "default_grid": DEFAULT_K_VALUES,
        "hadley_ls_k_exact": round(fit_quadratic_coefficient(), 6),
        "deployed_k": deployed_k,
        "n_xc_with_weather": int(len(df)),
        "comparisons": rows,
        "ranking_by_hadley_linear_rmse": [r["k"] for r in ranked],
        "best_hadley_linear_rmse_k": best["k"],
        "deployed_vs_best": {
            "deployed_k": deployed_k,
            "best_k": best["k"],
            "same": best["k"] == deployed_k,
            "linear_rmse_delta_pp": round(
                deployed_row["literature"]["rmse_vs_hadley_linear_101_180_pp"]
                - best["literature"]["rmse_vs_hadley_linear_101_180_pp"],
                4,
            ),
        },
        "k_0.0016_vs_deployed": (
            {
                "linear_rmse_improvement_pp": round(
                    deployed_row["literature"]["rmse_vs_hadley_linear_101_180_pp"]
                    - k16_row["literature"]["rmse_vs_hadley_linear_101_180_pp"],
                    4,
                ),
                "pct_H_in_band_gain": round(
                    k16_row["literature"]["pct_integer_H_inside_hadley_band"]
                    - deployed_row["literature"]["pct_integer_H_inside_hadley_band"],
                    1,
                ),
            }
            if k16_row
            else None
        ),
        "hadley_ls_vs_deployed": vs_deployed,
        "recommendation": _recommendation(rows, deployed_k, best, k16_row, hadley_ls_row),
    }


def _recommendation(
    rows: list[dict],
    deployed_k: float,
    best: dict,
    k16_row: dict | None,
    hadley_ls_row: dict | None,
) -> str:
    dep = next(r for r in rows if r["is_deployed"])
    if abs(best["k"] - deployed_k) < 1e-12:
        return f"Keep k={deployed_k}: best Hadley linear RMSE in this grid."

    parts = [
        f"Best Hadley linear RMSE: k={best['k']} "
        f"({best['literature']['rmse_vs_hadley_linear_101_180_pp']} pp).",
        f"Deployed k={deployed_k}: {dep['literature']['rmse_vs_hadley_linear_101_180_pp']} pp.",
    ]
    if k16_row and not k16_row["is_deployed"]:
        d = (
            dep["literature"]["rmse_vs_hadley_linear_101_180_pp"]
            - k16_row["literature"]["rmse_vs_hadley_linear_101_180_pp"]
        )
        parts.append(
            f"k=0.0016 vs 0.0015: {d:.3f} pp lower linear RMSE, "
            f"{k16_row['literature']['pct_integer_H_inside_hadley_band']:.0f}% vs "
            f"{dep['literature']['pct_integer_H_inside_hadley_band']:.0f}% of H in Hadley band — optional upgrade."
        )
    if hadley_ls_row and not hadley_ls_row["is_deployed"]:
        parts.append(
            f"Hadley LS optimum k≈{hadley_ls_row['k']:.4f} is marginally better than both; "
            "difference from 0.0015/0.0016 is small for CIKM."
        )
    parts.append("Empirical r(adj,H) changes little across this grid; choose on Hadley fidelity.")
    return " ".join(parts)


def print_markdown_table(result: dict) -> None:
    print("\n| k | Hadley linear RMSE | Midpoint RMSE | % H in band | Mean slow. hot | r(adj,H) hot |")
    print("|---|-------------------|---------------|-------------|----------------|--------------|")
    for r in sorted(result["comparisons"], key=lambda x: x["k"]):
        lit = r["literature"]
        emp = r["empirical"]
        mark = ""
        if r["is_deployed"]:
            mark = " **"
        elif r["is_hadley_ls"]:
            mark = " *"
        k_str = f"{r['k']:.6f}".rstrip("0").rstrip(".")
        print(
            f"| {k_str}{mark} | {lit['rmse_vs_hadley_linear_101_180_pp']} | "
            f"{lit['rmse_vs_band_midpoints_pp']} | {lit['pct_integer_H_inside_hadley_band']} | "
            f"{emp['mean_slowdown_pct_when_hot']} | {emp['corr_after_heat_vs_H_hot']} |"
        )
    print("\n** = deployed; * = Hadley LS optimum\n")
    print(result["recommendation"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare heat coefficient k candidates")
    parser.add_argument(
        "--k",
        type=float,
        nargs="*",
        default=None,
        help="Coefficients to compare (default: built-in grid incl. 0.0015 and 0.0016)",
    )
    parser.add_argument("--write", type=Path, default=None, help="JSON output path")
    parser.add_argument("--markdown", action="store_true", help="Print comparison table")
    args = parser.parse_args()

    k_values = args.k if args.k else DEFAULT_K_VALUES
    tables = load_tables()
    df = _build_xc_weather_frame(tables)
    if len(df) < 100:
        print(json.dumps({"error": "insufficient data", "n": len(df)}))
        sys.exit(1)

    result = compare_k_values(df, k_values)
    print(json.dumps(result, indent=2))

    if args.markdown:
        print_markdown_table(result)

    out = args.write or (RESULTS / "heat_compare_k.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
