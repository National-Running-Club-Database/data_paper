#!/usr/bin/env python3
"""Run all NRCD dataset statistics and validations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as scripts/run_all.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS
from load_data import build_results_frame, load_tables
from stats_composition import composition_summary
from stats_coverage import coverage_by_sport_era, gender_by_sport_era
from stats_event_breakdown import event_breakdown
from validation_heat import validate_heat_adjustment
from validation_improvement import improvement_summary
from stats_longitudinal import longitudinal_depth_summary
from validation_k_sensitivity import k_sensitivity_summary
from validation_variance import within_athlete_variance
from validation_variance_asymmetry import variance_asymmetry_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="Skip diagnostics not cited in cikm.tex (variance_asymmetry, ML R²).",
    )
    parser.add_argument(
        "--skip-k-sensitivity",
        action="store_true",
        help="Skip k grid (~5 min); paper Table ksens needs this unless already in JSON.",
    )
    parser.add_argument(
        "--run-ml",
        action="store_true",
        help="Run illustrative RF R² check (requires sklearn; not in dataset_stats.json).",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    tables = load_tables()
    df = build_results_frame(tables)

    payload = {
        "composition": composition_summary(tables, df),
        "coverage_by_sport_era": coverage_by_sport_era(df),
        "gender_by_sport_era": gender_by_sport_era(df),
        "event_breakdown": event_breakdown(df),
        "variance_validation": within_athlete_variance(tables),
        "heat_validation": validate_heat_adjustment(tables),
        "improvement_demo": improvement_summary(tables),
        "longitudinal_depth": longitudinal_depth_summary(df),
    }

    if not args.paper_only:
        payload["variance_asymmetry"] = variance_asymmetry_summary(tables)

    if not args.skip_k_sensitivity:
        payload["k_sensitivity"] = k_sensitivity_summary(tables)

    out_json = RESULTS / "dataset_stats.json"
    out_json.write_text(json.dumps(payload, indent=2))

    if args.run_ml or (not args.paper_only):
        try:
            from validation_ml_r2 import main as ml_r2_main

            ml_r2_main()
        except ImportError:
            if args.run_ml:
                print("ML check skipped: sklearn not installed", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out_json} ({elapsed:.1f}s)")

    heat = payload["heat_validation"]
    passed = heat.get("empirical", {}).get("validation_passed")
    print(f"\nHeat validation passed: {passed}")


if __name__ == "__main__":
    main()
