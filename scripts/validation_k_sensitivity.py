"""Sensitivity of XC validation metrics to heat quadratic coefficient k."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import HEAT_QUADRATIC_COEFF
from standardization import converted_only_xc, standardize_xc_row
from validation_improvement import _median_improvement_by_gender


DEFAULT_K_VALUES = (0.001, 0.0016, 0.002)


def _xc_performance_records(tables: dict, heat_k: float, min_meets: int = 3) -> pd.DataFrame:
    meet = tables["meet"]
    result = tables["result"]
    athlete = tables["athlete"]
    running_event = tables["running_event"]
    course_details = tables["course_details"]

    xc_meets = set(meet.loc[meet["sport_id"] == 1, "meet_id"])
    df = result[result["meet_id"].isin(xc_meets)].copy()
    df = df.merge(athlete[["athlete_id", "gender"]], on="athlete_id")
    df = df.merge(running_event, on="running_event_id")
    from schema import meet_altitude_column

    alt_col = meet_altitude_column(meet)
    df = df.merge(
        meet[["meet_id", "start_date", alt_col]].rename(columns={alt_col: "altitude"}),
        on="meet_id",
    )
    df["season"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year
    nationals = meet.set_index("meet_id")["nationals"].astype(bool)
    df = df[~df["meet_id"].map(nationals).fillna(False)]

    records = []
    for _, row in df.iterrows():
        raw, std = standardize_xc_row(row, course_details, heat_k=heat_k)
        if np.isfinite(raw) and np.isfinite(std):
            records.append(
                {
                    "athlete_id": row["athlete_id"],
                    "season": row["season"],
                    "meet_id": row["meet_id"],
                    "gender": row["gender"],
                    "raw": raw,
                    "standardized": std,
                }
            )
    return pd.DataFrame(records).dropna(subset=["season"])


def _variance_by_gender(perf: pd.DataFrame, min_meets: int) -> dict[str, dict]:
    out = {}
    for gender, label in [("F", "women"), ("M", "men")]:
        g = perf[perf["gender"] == gender]
        athlete_season_stats = []
        for (_, _), grp in g.groupby(["athlete_id", "season"]):
            if grp["meet_id"].nunique() < min_meets:
                continue
            athlete_season_stats.append(
                {"raw_sd": grp["raw"].std(), "std_sd": grp["standardized"].std()}
            )
        if not athlete_season_stats:
            out[label] = {}
            continue
        stats_df = pd.DataFrame(athlete_season_stats)
        med_raw = float(stats_df["raw_sd"].median())
        med_std = float(stats_df["std_sd"].median())
        out[label] = {
            "n_athlete_seasons": len(stats_df),
            "median_raw_sd_sec": round(med_raw, 2),
            "median_std_sd_sec": round(med_std, 2),
            "pct_reduction_std_vs_raw": round(100 * (1 - med_std / med_raw), 1),
        }
    return out


def _xc_improvements_k(tables: dict, heat_k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    meet = tables["meet"]
    result = tables["result"]
    athlete = tables["athlete"]
    running_event = tables["running_event"]
    course_details = tables["course_details"]

    xc_meets = set(
        meet.loc[(meet["sport_id"] == 1) & (~meet["nationals"].astype(bool)), "meet_id"]
    )
    df = result[result["meet_id"].isin(xc_meets)].copy()
    df = df.merge(athlete[["athlete_id", "gender"]], on="athlete_id")
    df = df.merge(running_event, on="running_event_id")
    from schema import meet_altitude_column

    alt_col = meet_altitude_column(meet)
    df = df.merge(
        meet[["meet_id", "start_date", alt_col]].rename(columns={alt_col: "altitude"}),
        on="meet_id",
    )
    df["race_date"] = pd.to_datetime(df["start_date"], errors="coerce")

    conv_rows, std_rows = [], []
    for _, row in df.iterrows():
        conv = converted_only_xc(row, course_details)
        _, std = standardize_xc_row(row, course_details, heat_k=heat_k)
        if np.isfinite(conv):
            conv_rows.append(
                {
                    "athlete_id": row["athlete_id"],
                    "gender": row["gender"],
                    "season": row["race_date"].year,
                    "race_date": row["race_date"],
                    "time": conv,
                }
            )
        if np.isfinite(std):
            std_rows.append(
                {
                    "athlete_id": row["athlete_id"],
                    "gender": row["gender"],
                    "season": row["race_date"].year,
                    "race_date": row["race_date"],
                    "time": std,
                }
            )
    return (
        pd.DataFrame(conv_rows).dropna(subset=["race_date"]),
        pd.DataFrame(std_rows).dropna(subset=["race_date"]),
    )


def _improvement_inflation(tables: dict, heat_k: float) -> dict[str, dict]:
    conv, std = _xc_improvements_k(tables, heat_k)
    by_mode = {
        "converted_only": _median_improvement_by_gender(conv),
        "fully_standardized": _median_improvement_by_gender(std),
    }
    conv_imp, std_imp = {}, {}
    for df, store in ((conv, conv_imp), (std, std_imp)):
        for (aid, season, gender), grp in df.groupby(["athlete_id", "season", "gender"]):
            if len(grp) < 2:
                continue
            grp = grp.sort_values("race_date")
            store[(aid, season, gender)] = float(
                grp.iloc[0]["time"] - grp.iloc[1:]["time"].min()
            )
    out = {}
    for gender, label in [("M", "men"), ("F", "women")]:
        keys = [k for k in conv_imp if k[2] == gender and k in std_imp]
        c_arr = np.array([conv_imp[k] for k in keys])
        s_arr = np.array([std_imp[k] for k in keys])
        c = by_mode["converted_only"][label]["median_improvement_sec"]
        s = by_mode["fully_standardized"][label]["median_improvement_sec"]
        if c is not None and s is not None and s > 0 and len(c_arr) > 0:
            ratio = c / s
            out[label] = {
                "median_conv_sec": c,
                "median_std_sec": s,
                "pct_inflation_vs_full": round(100 * (ratio - 1), 0),
            }
        else:
            out[label] = {}
    return out


def k_sensitivity_summary(
    tables: dict,
    k_values: tuple[float, ...] | None = None,
    min_meets: int = 3,
) -> dict:
    """Variance reduction and improvement inflation vs. heat k (converted-only unchanged)."""
    k_values = k_values or DEFAULT_K_VALUES
    conv, _ = _xc_improvements_k(tables, heat_k=HEAT_QUADRATIC_COEFF)
    conv_by_gender = _median_improvement_by_gender(conv)

    by_k = []
    for k in k_values:
        perf = _xc_performance_records(tables, heat_k=k, min_meets=min_meets)
        _, std = _xc_improvements_k(tables, heat_k=k)
        std_by_gender = _median_improvement_by_gender(std)
        inflation = {}
        for label in ("men", "women"):
            c = conv_by_gender.get(label, {}).get("median_improvement_sec")
            s = std_by_gender.get(label, {}).get("median_improvement_sec")
            if c is not None and s is not None and s > 0:
                inflation[label] = {
                    "median_conv_sec": c,
                    "median_std_sec": s,
                    "pct_inflation_vs_full": round(100 * (c / s - 1), 0),
                }
            else:
                inflation[label] = {}
        by_k.append(
            {
                "k": k,
                "is_deployed": abs(k - HEAT_QUADRATIC_COEFF) < 1e-12,
                "variance_by_gender": _variance_by_gender(perf, min_meets),
                "improvement_inflation": inflation,
            }
        )

    return {"k_values": list(k_values), "by_k": by_k}
