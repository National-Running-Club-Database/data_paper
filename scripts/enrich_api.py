#!/usr/bin/env python3
"""Backfill meet altitude and course weather via nrcd.enrich (optional parallelism).

Processes each lookup as it completes so one failure does not stop the run.
In-process TTL cache dedupes repeated city/state within the same batch only.

Examples
--------
  export NRCD_OPENWEATHER_API_KEY=...
  export NRCD_TIMEZONE_API_KEY=...

  # Sequential (default) — gentle on TimeZoneDB
  python scripts/enrich_api.py --task altitude --limit 50

  # Parallel geocode/altitude (cache still dedupes city/state)
  python scripts/enrich_api.py --task altitude --limit 200 --parallel 10

  # Weather: parallel workers, results reported one-by-one as each finishes
  python scripts/enrich_api.py --task weather --limit 100 --parallel 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA, RESULTS  # noqa: E402

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas", file=sys.stderr)
    sys.exit(1)

try:
    from nrcd.enrich import (  # noqa: E402
        ApiUsage,
        EnrichConfig,
        api_keys_from_env,
        cache_stats,
        clear_enrich_cache,
        fetch_weather,
        lookup_altitude_ft,
        reset_throttle_state,
    )
    from nrcd.enrich.batch import EnrichJob, JobResult, run_enrich_jobs
except ImportError:
    print(
        "nrcd package required: pip install nrcd[apis] (separate from this paper repo)",
        file=sys.stderr,
    )
    sys.exit(1)


def _parse_date(val) -> dt.date | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return pd.Timestamp(val).date()
    except Exception:
        return None


def _parse_time(val) -> dt.time | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        ts = pd.Timestamp(str(val))
        return ts.time()
    except Exception:
        return None


def _load_tables():
    meet = pd.read_csv(DATA / "meet.csv")
    sport = pd.read_csv(DATA / "sport.csv")
    course_details = pd.read_csv(DATA / "course_details.csv", low_memory=False)
    return meet, sport, course_details


def _meet_altitude_jobs(meet: pd.DataFrame, *, limit: int, cfg: EnrichConfig) -> list[EnrichJob]:
    alt_col = "altitude" if "altitude" in meet.columns else "elevation"
    missing = meet[meet[alt_col].isna()].copy()
    missing["meet_city"] = missing["meet_city"].fillna("").astype(str).str.strip()
    missing["meet_state"] = missing["meet_state"].fillna("").astype(str).str.strip()
    missing = missing[(missing["meet_city"] != "") & (missing["meet_state"] != "")]
    missing = missing.head(limit)

    jobs: list[EnrichJob] = []

    lat_col = "meet_latitude" if "meet_latitude" in missing.columns else None
    lon_col = "meet_longitude" if "meet_longitude" in missing.columns else None

    def _make(mid: int, city: str, state: str, lat=None, lon=None):
        def run():
            usage = ApiUsage()
            ft = lookup_altitude_ft(
                city,
                state,
                config=cfg,
                lat=lat,
                lon=lon,
                usage=usage,
            )
            if ft is None:
                raise RuntimeError("Geocode or USGS returned no altitude")
            return {
                "meet_id": int(mid),
                "altitude": int(ft),
                "city": city,
                "state": state,
                "api_usage": usage.to_dict(),
            }

        return run

    for row in missing.itertuples(index=False):
        lat = getattr(row, lat_col) if lat_col else None
        lon = getattr(row, lon_col) if lon_col else None
        if lat is not None and lon is not None and pd.notna(lat) and pd.notna(lon):
            lat, lon = float(lat), float(lon)
        else:
            lat, lon = None, None
        jobs.append(
            EnrichJob(
                job_id=f"meet:{row.meet_id}",
                run=_make(row.meet_id, row.meet_city, row.meet_state, lat, lon),
            )
        )
    return jobs


def _weather_jobs(
    meet: pd.DataFrame,
    sport: pd.DataFrame,
    course_details: pd.DataFrame,
    *,
    limit: int,
    cfg: EnrichConfig,
) -> list[EnrichJob]:
    cd = course_details[
        course_details["temperature"].isna() & course_details["dew_point"].isna()
    ].copy()
    cd = cd[cd["date_of_event"].notna() & cd["time_of_event"].notna()]
    cd = cd.head(limit * 4)

    m = meet.merge(sport[["sport_id", "sport_name"]], on="sport_id", how="left")
    meet_cols = ["meet_id", "meet_city", "meet_state", "sport_name"]
    for col in ("meet_latitude", "meet_longitude", "meet_timezone"):
        if col in m.columns:
            meet_cols.append(col)
    m = m[meet_cols]
    cd = cd.merge(m, on="meet_id", how="inner")
    cd["meet_city"] = cd["meet_city"].fillna("").astype(str).str.strip()
    cd["meet_state"] = cd["meet_state"].fillna("").astype(str).str.strip()
    cd = cd[(cd["meet_city"] != "") & (cd["meet_state"] != "")]
    cd = cd[~cd["sport_name"].fillna("").str.contains("Indoor Track", case=False, na=False)]

    jobs: list[EnrichJob] = []
    count = 0

    for row in cd.itertuples(index=False):
        if count >= limit:
            break
        event_date = _parse_date(row.date_of_event)
        event_time = _parse_time(row.time_of_event)
        if event_date is None or event_time is None:
            continue
        cid = int(row.course_details_id)
        city, state = row.meet_city, row.meet_state

        lat = getattr(row, "meet_latitude", None)
        lon = getattr(row, "meet_longitude", None)
        tz = getattr(row, "meet_timezone", None)
        if lat is not None and lon is not None and pd.notna(lat) and pd.notna(lon):
            lat, lon = float(lat), float(lon)
        else:
            lat, lon = None, None
        tz_name = str(tz).strip() if tz is not None and pd.notna(tz) and str(tz).strip() else None

        def run(
            cid=cid,
            city=city,
            state=state,
            event_date=event_date,
            event_time=event_time,
            lat=lat,
            lon=lon,
            tz_name=tz_name,
        ):
            usage = ApiUsage()
            wx = fetch_weather(
                city,
                state,
                event_date,
                event_time,
                config=cfg,
                lat=lat,
                lon=lon,
                timezone_name=tz_name,
                usage=usage,
            )
            if wx is None:
                raise RuntimeError("Geocode, timezone, or OpenWeather returned no data")
            out = wx.as_course_details_dict()
            out["course_details_id"] = cid
            out["city"] = city
            out["state"] = state
            out["api_usage"] = usage.to_dict()
            return out

        jobs.append(EnrichJob(job_id=f"cd:{cid}", run=run))
        count += 1

    return jobs


def _print_result(res: JobResult) -> None:
    if res.ok:
        print(f"  ok {res.job_id}")
    else:
        print(f"  FAIL {res.job_id}: {res.error}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("altitude", "weather", "both"),
        default="altitude",
        help="Meet altitude (USGS), course weather (OpenWeather), or both",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max rows per task")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Worker threads (1=sequential). Results still handled one-by-one as each finishes.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear in-process enrich cache before run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "enrich_api_report.json",
        help="JSON report path",
    )
    parser.add_argument(
        "--updates-csv",
        type=Path,
        default=None,
        help="Write successful lookups to CSV (altitude or weather columns)",
    )
    parser.add_argument("--openweather-key", default=None)
    parser.add_argument("--timezone-key", default=None)
    args = parser.parse_args()

    cfg = api_keys_from_env()
    if args.openweather_key:
        cfg = EnrichConfig(
            openweather_api_key=args.openweather_key,
            timezone_api_key=args.timezone_key or cfg.timezone_api_key,
            timezone_min_interval_sec=cfg.timezone_min_interval_sec,
            openweather_min_interval_sec=cfg.openweather_min_interval_sec,
            cache_enabled=cfg.cache_enabled,
        )
    elif args.timezone_key:
        cfg = EnrichConfig(
            openweather_api_key=cfg.openweather_api_key,
            timezone_api_key=args.timezone_key,
        )

    if not cfg.openweather_api_key and args.task in ("altitude", "weather", "both"):
        print("Set NRCD_OPENWEATHER_API_KEY or --openweather-key", file=sys.stderr)
        sys.exit(1)
    if args.task in ("weather", "both") and not cfg.timezone_api_key:
        print("Set NRCD_TIMEZONE_API_KEY or --timezone-key for weather", file=sys.stderr)
        sys.exit(1)

    if args.clear_cache:
        clear_enrich_cache()
        reset_throttle_state()

    meet, sport, course_details = _load_tables()
    RESULTS.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "parallel": max(1, args.parallel),
        "tasks": {},
        "failures": [],
        "cache": {},
        "api_usage_total": ApiUsage().to_dict(),
    }
    usage_total = ApiUsage()
    updates: list[dict] = []

    def on_result(res: JobResult) -> None:
        _print_result(res)
        if not res.ok:
            report["failures"].append(
                {"job_id": res.job_id, "error": res.error},
            )
        elif res.value:
            row = dict(res.value)
            job_usage = row.pop("api_usage", None)
            if job_usage:
                usage_total.add(ApiUsage.from_dict(job_usage))
            updates.append(row)

    if args.task in ("altitude", "both"):
        jobs = _meet_altitude_jobs(meet, limit=args.limit, cfg=cfg)
        print(f"Altitude jobs: {len(jobs)} (parallel={args.parallel})")
        results = run_enrich_jobs(jobs, parallel=args.parallel, on_result=on_result)
        report["tasks"]["altitude"] = {
            "queued": len(jobs),
            "ok": sum(1 for r in results if r.ok),
            "failed": sum(1 for r in results if not r.ok),
        }

    if args.task in ("weather", "both"):
        jobs = _weather_jobs(meet, sport, course_details, limit=args.limit, cfg=cfg)
        print(f"Weather jobs: {len(jobs)} (parallel={args.parallel})")
        results = run_enrich_jobs(jobs, parallel=args.parallel, on_result=on_result)
        report["tasks"]["weather"] = {
            "queued": len(jobs),
            "ok": sum(1 for r in results if r.ok),
            "failed": sum(1 for r in results if not r.ok),
        }

    report["cache"] = cache_stats()
    report["api_usage_total"] = usage_total.to_dict()
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Cache stats: {report['cache']}")
    print(f"API calls (HTTP, cache misses): {report['api_usage_total']}")

    if args.updates_csv and updates:
        pd.DataFrame(updates).to_csv(args.updates_csv, index=False)
        print(f"Wrote {len(updates)} successful rows to {args.updates_csv}")


if __name__ == "__main__":
    main()
