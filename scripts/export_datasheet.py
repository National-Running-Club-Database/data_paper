#!/usr/bin/env python3
"""
Export a Datasheets-style markdown summary from the local NRCD export.

Usage:
  python scripts/export_datasheet.py
  python scripts/run_all.py   # run first for freshest dataset_stats.json
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMPREHENSIVE_FROM, DATA, RESULTS, ROOT
from load_data import build_results_frame, load_tables
from stats_composition import composition_summary

OUTPUT = ROOT / "docs" / "NRCD_DATASHEET.md"


def _load_stats() -> dict:
    p = RESULTS / "dataset_stats.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def build_datasheet_md() -> str:
    tables = load_tables()
    comp = composition_summary(tables, build_results_frame(tables))
    stats = _load_stats()

    lines = [
        "# NRCD Public Export Datasheet",
        "",
        f"**Snapshot date:** {date.today().isoformat()}  ",
        "**Version:** May 2026 public approved export  ",
        "**License:** See [dataset repository](https://github.com/National-Running-Club-Database/national_running_club_database_public_dataset) and [Zenodo](https://zenodo.org/records/17917357).",
        "",
        "## Motivation",
        "Provide the first large-scale, relational, anonymized corpus of U.S. collegiate club running results with course and weather metadata for research on performance, fairness, and environmental confounds.",
        "",
        "## Composition",
        f"- **Results:** {comp['total_results']:,}",
        f"- **Athletes:** {comp['total_athletes']:,}",
        f"- **Meets:** {comp['total_meets']:,}",
        f"- **Teams:** {comp['total_teams']:,}",
        f"- **Sports:** Cross Country, Indoor Track, Outdoor Track, Road Race",
        f"- **Date range:** {comp.get('date_min', '?')} to {comp.get('date_max', '?')}",
        "",
        "## Collection process",
        "- Open submission on nationalrunningclubdatabase.com",
        "- NIRCA expert administrators approve meets and results",
        "- Human review for duplicate or similar athlete names on a team (>90% similarity or shared last name)",
        "- Sources: member teams, permitted public meet pages, authorized uploads",
        "- Weather on course_details via OpenWeatherMap at meet time (where recorded)",
        "",
        "## Preprocessing / release filtering",
        "- Only `approved=True` meets and results",
        "- PII removed (names, profile URLs); stable pseudonymous athlete_id",
        "- Normalized CSV tables + optional denormalized joined view in upstream repo",
        "",
        "## Recommended uses",
        "- Longitudinal athlete and team modeling",
        "- Environmental confounder studies (weather, elevation, course length)",
        "- Gender-equity and participation analyses",
        "- Benchmarking ML with temporal splits",
        "",
        "## Uses to avoid",
        "- Re-identification of athletes",
        "- Cross-gender comparison of raw or pooled seconds in XC (use 6 km / 8 km standardized times and within-gender metrics)",
        "- Treating pre-August 2023 rows as having complete weather metadata",
        "",
        "## Maintenance",
        "- Annual versioned Zenodo snapshots planned",
        "- Live database continues on the project website",
        "",
        "## Files (local `data/`)",
    ]
    for f in sorted(DATA.glob("*.csv")):
        lines.append(f"- `{f.name}`")
    lines.append("")
    lines.append("## Coverage regimes")
    lines.append(f"- **Comprehensive:** meet date >= {COMPREHENSIVE_FROM}")
    lines.append(f"- **Historical:** earlier meets (sparser metadata)")
    if stats.get("coverage_by_sport_era"):
        lines.append("")
        lines.append("### Metadata coverage (from analysis)")
        for row in stats["coverage_by_sport_era"]:
            lines.append(
                f"- {row['sport']} / {row['era']}: {row['results']:,} results, "
                f"{row.get('weather_pct', row.get('metadata_pct', 0))}% weather, "
                f"{row.get('meet_altitude_pct', 0)}% meet altitude, "
                f"{row.get('course_features_pct', row.get('grade_pct', 0))}% course features"
            )
    lines.append("")
    lines.append("## Reproduce paper statistics")
    lines.append("```bash")
    lines.append("pip install -r requirements.txt")
    lines.append("# place CSVs in data/")
    lines.append("python scripts/run_all.py")
    lines.append("python scripts/example_load_dataset.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    md = build_datasheet_md()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(md)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
