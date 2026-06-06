#!/usr/bin/env python3
"""
Minimal NRCD loading example for new users (CIKM utility / datasheet companion).

Usage:
  python scripts/example_load_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMPREHENSIVE_FROM
from load_data import build_results_frame, load_tables
from standardization import standardize_xc_row


def main() -> None:
    tables = load_tables()
    df = build_results_frame(tables)

    print("=== NRCD quick load ===\n")
    print(f"Approved results (merged frame): {len(df):,}")
    print(f"Sports: {df['sport_name'].value_counts().to_dict()}")
    print(f"Comprehensive era (>= {COMPREHENSIVE_FROM}): {df['era'].value_counts().to_dict()}")
    print(f"Gender: {df['gender'].value_counts().to_dict()}")
    print(f"Results with weather metadata: {int(df['has_weather_meta'].sum()):,}")

    xc = df[df["sport_name"] == "Cross Country"].head(1)
    if len(xc):
        row = xc.iloc[0]
        raw, std = standardize_xc_row(row, tables["course_details"])
        print(f"\nExample XC standardization (one row):")
        print(f"  gender={row['gender']}  event={row.get('event_name', '?')}")
        print(f"  raw time (s): {raw:.1f}  standardized (s): {std:.1f}")

    print("\nNext: python scripts/run_all.py  # full paper statistics")


if __name__ == "__main__":
    main()
