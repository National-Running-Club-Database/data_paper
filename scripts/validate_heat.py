#!/usr/bin/env python3
"""Validate heat (temperature + dew point) adjustment formula and empirical effect."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS
from load_data import load_tables
from validation_heat import validate_heat_adjustment


def main() -> None:
    tables = load_tables()
    result = validate_heat_adjustment(tables)
    print(json.dumps(result, indent=2))

    out = RESULTS / "heat_validation.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")

    pf = result.get("piecewise_fit", {})
    if pf:
        print(
            f"\nPiecewise fit: k_deployed={pf.get('deployed_coefficient_k')}, "
            f"k_LS={pf.get('least_squares_k_vs_band_midpoints')}, "
            f"RMSE={pf.get('rmse_pct_vs_band_midpoints')} pp, "
            f"within_band={pf.get('integer_H_101_to_180', {}).get('pct_within_hadley_band')}%",
            file=sys.stderr,
        )

    passed = result.get("empirical", {}).get("validation_passed")
    if not result.get("unit_tests", {}).get("monotone_above_100"):
        sys.exit(1)
    if passed is False:
        print("Warning: hot-race median time was not reduced after adjustment.", file=sys.stderr)


if __name__ == "__main__":
    main()
