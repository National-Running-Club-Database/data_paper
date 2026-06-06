#!/usr/bin/env python3
"""
Verify local NRCD export against paper claims (FAIR / availability checklist).

Usage:
  python scripts/fair_checklist.py
  python scripts/fair_checklist.py --write results/fair_checklist.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA, RESULTS, ROOT
from load_data import build_results_frame, load_tables
from stats_composition import composition_summary

EXPECTED_CSV = [
    "athlete.csv",
    "athlete_team_association.csv",
    "course_details.csv",
    "meet.csv",
    "result.csv",
    "running_event.csv",
    "sport.csv",
    "team.csv",
]

ZENODO_URL = "https://zenodo.org/records/17917357"
DATASET_REPO = "https://github.com/National-Running-Club-Database/national_running_club_database_public_dataset"


def run_checklist() -> dict:
    files = {}
    for name in EXPECTED_CSV:
        p = DATA / name
        files[name] = {"exists": p.exists(), "path": str(p)}

    missing = [n for n, v in files.items() if not v["exists"]]
    tables = load_tables() if not missing else {}
    composition = composition_summary(tables, build_results_frame(tables)) if tables else {}

    analysis_readme = (ROOT / "README.md").exists()
    datasheet = (ROOT / "docs" / "NRCD_DATASHEET.md").exists()

    return {
        "data_directory": str(DATA),
        "expected_csv_files": files,
        "missing_files": missing,
        "load_ok": len(missing) == 0,
        "composition": composition,
        "urls": {
            "zenodo": ZENODO_URL,
            "dataset_github": DATASET_REPO,
            "analysis_github": "https://github.com/National-Running-Club-Database/data_paper",
        },
        "local_docs": {
            "root_readme": analysis_readme,
            "datasheet_md": datasheet,
            "results_dataset_stats": (RESULTS / "dataset_stats.json").exists(),
        },
        "fair_summary": {
            "findable": bool(ZENODO_URL),
            "accessible": len(missing) == 0,
            "interoperable": "CSV",
            "reusable": "IRB non-human-subjects; PII removed in public export",
        },
        "check_passed": len(missing) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NRCD FAIR / availability checklist")
    parser.add_argument("--write", type=Path, default=RESULTS / "fair_checklist.json")
    args = parser.parse_args()

    out = run_checklist()
    print(json.dumps(out, indent=2))

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.write}", file=sys.stderr)

    if not out["check_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
