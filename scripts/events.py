"""Classify running_event names into event categories."""

from __future__ import annotations

import math
import re


def event_category(event_name: str) -> str:
    """Return one of: sprint, mid_distance, distance, relay, field.

  Rules: sprint <= 400 m plus hurdles; mid-distance 401--1000 m; distance > 1000 m
  plus steeplechase; relay and field by keyword. Unmatched names return ``other``.
    """
    n = str(event_name).lower()

    relay_kw = ("4x", "relay", "smr", "dmr", "swedish")
    if any(k in n for k in relay_kw):
        return "relay"

    field_kw = ("jump", "put", "throw", "vault", "discus", "javelin", "hammer")
    if any(k in n for k in field_kw):
        return "field"

    if "hurdle" in n:
        return "sprint"

    if "steeple" in n:
        return "distance"

    if any(k in n for k in ("marathon", "half marathon", "10 mile", "14 mile")):
        return "distance"
    if re.search(r"\d+\s*mile|4 mile|5 mile|2 mile", n):
        return "distance"

    dist_m = re.search(r"(\d+)m", n)
    if dist_m:
        d = int(dist_m.group(1))
        if d <= 400:
            return "sprint"
        if d <= 1000:
            return "mid_distance"
        return "distance"

    if n in ("mile", "2 mile", "4 mile", "5 mile"):
        return "distance"

    return "other"


def parse_event_distance_m(event_name: str | None) -> float | None:
    """Distance in meters from event label (e.g. ``8000m``, ``110m Hurdles``)."""
    if event_name is None:
        return None
    if isinstance(event_name, float) and not math.isfinite(event_name):
        return None
    event_name = str(event_name)
    lower = event_name.lower()
    if event_name.endswith("m"):
        try:
            return float(event_name.replace("m", "").strip())
        except ValueError:
            pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*m", event_name, re.I)
    if m:
        return float(m.group(1))
    if lower in ("mile", "1 mile", "1600m"):
        return 1609.34
    if lower == "4 mile":
        return 4 * 1609.34
    if lower == "5 mile":
        return 5 * 1609.34
    if lower == "2 mile":
        return 2 * 1609.34
    return None


def applies_altitude_conversion(event_name: str | None) -> bool:
    if event_name is None:
        return False
    cat = event_category(str(event_name))
    if cat in ("field", "relay", "other"):
        return False
    return parse_event_distance_m(event_name) is not None
