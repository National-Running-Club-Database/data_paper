"""Within-season improvement: converted-only vs full standardization (formula validation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from progress import iter_progress
from standardization import compute_xc_times
from xc_frame import build_xc_frame


def _improvement_times(
    tables: dict, *, min_races: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (converted-only, standardized) athlete-season race clocks in one pass."""
    timed = compute_xc_times(build_xc_frame(tables, exclude_nationals=True))
    base_cols = ["athlete_id", "gender", "season", "race_date", "meet_id"]

    def _time_frame(col: str) -> pd.DataFrame:
        out = timed.loc[np.isfinite(timed[col]), base_cols + [col]].copy()
        out = out.rename(columns={col: "time"}).dropna(subset=["race_date"])
        if min_races > 1:
            counts = out.groupby(["athlete_id", "season", "gender"])["time"].transform("size")
            out = out[counts >= min_races]
        return out

    return _time_frame("converted_sec"), _time_frame("standardized_sec")


def _improvement_frame(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _, gender), grp in perf.groupby(["athlete_id", "season", "gender"]):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("race_date")
        rows.append(
            {
                "gender": gender,
                "improvement_sec": grp.iloc[0]["time"] - grp.iloc[1:]["time"].min(),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_median_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    meds = []
    n = len(values)
    for _ in iter_progress(range(n_boot), desc="Bootstrap median", unit="rep", leave=False):
        sample = values[rng.integers(0, n, size=n)]
        meds.append(float(np.median(sample)))
    meds_arr = np.array(meds)
    return (
        float(np.median(values)),
        float(np.percentile(meds_arr, 2.5)),
        float(np.percentile(meds_arr, 97.5)),
    )


def _median_improvement_by_gender(perf: pd.DataFrame, bootstrap: bool = False) -> dict[str, dict]:
    imp = _improvement_frame(perf)
    out = {}
    for gender, label in [("M", "men"), ("F", "women")]:
        g = imp[imp["gender"] == gender]
        if len(g) == 0:
            out[label] = {"n_athlete_seasons": 0, "median_improvement_sec": None}
            continue
        vals = g["improvement_sec"].to_numpy()
        med, lo, hi = _bootstrap_median_ci(vals) if bootstrap else (float(np.median(vals)), None, None)
        entry = {
            "n_athlete_seasons": int(len(g)),
            "median_improvement_sec": round(med, 1),
        }
        if bootstrap:
            entry["median_ci95_lo"] = round(lo, 1)
            entry["median_ci95_hi"] = round(hi, 1)
        out[label] = entry
    return out


def _bootstrap_ratio_ci(
    conv: np.ndarray, std: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap CI for median(conv)/median(std) on paired athlete-season indices."""
    rng = np.random.default_rng(seed)
    n = len(conv)
    ratios = []
    for _ in iter_progress(range(n_boot), desc="Bootstrap ratio", unit="rep", leave=False):
        idx = rng.integers(0, n, size=n)
        c_med = float(np.median(conv[idx]))
        s_med = float(np.median(std[idx]))
        if s_med > 0:
            ratios.append(c_med / s_med)
    ratios_arr = np.array(ratios)
    point = float(np.median(conv) / np.median(std)) if np.median(std) > 0 else np.nan
    return point, float(np.percentile(ratios_arr, 2.5)), float(np.percentile(ratios_arr, 97.5))


def improvement_summary(tables: dict, *, min_races: int = 2) -> dict:
    """Compare within-season improvement under converted-only vs full standardization."""
    conv, std = _improvement_times(tables, min_races=min_races)
    by_mode = {
        "converted_only": _median_improvement_by_gender(conv, bootstrap=True),
        "fully_standardized": _median_improvement_by_gender(std, bootstrap=True),
    }

    conv_imp = {}
    std_imp = {}
    for df, store in ((conv, conv_imp), (std, std_imp)):
        for (aid, season, gender), grp in df.groupby(["athlete_id", "season", "gender"]):
            if len(grp) < 2:
                continue
            grp = grp.sort_values("race_date")
            store[(aid, season, gender)] = float(
                grp.iloc[0]["time"] - grp.iloc[1:]["time"].min()
            )

    inflation = {}
    ratio_ci = {}
    for gender, label in [("M", "men"), ("F", "women")]:
        keys = [k for k in conv_imp if k[2] == gender and k in std_imp]
        c_arr = np.array([conv_imp[k] for k in keys])
        s_arr = np.array([std_imp[k] for k in keys])
        c = by_mode["converted_only"][label]["median_improvement_sec"]
        s = by_mode["fully_standardized"][label]["median_improvement_sec"]
        if c is not None and s is not None and s > 0 and len(c_arr) > 0:
            r, r_lo, r_hi = _bootstrap_ratio_ci(c_arr, s_arr)
            ratio = c / s
            inflation[label] = {
                "excess_median_sec": round(c - s, 1),
                "ratio_converted_to_standardized": round(ratio, 2),
                "pct_inflation_vs_full": round(100 * (ratio - 1), 1),
            }
            ratio_ci[label] = {
                "ratio_point": round(r, 2),
                "ratio_ci95_lo": round(r_lo, 2),
                "ratio_ci95_hi": round(r_hi, 2),
            }
        else:
            inflation[label] = None
            ratio_ci[label] = None

    pct_positive = {
        "converted_only": _pct_positive_improvement(conv),
        "fully_standardized": _pct_positive_improvement(std),
    }

    return {
        "by_mode": by_mode,
        "converted_inflation_vs_standardized": inflation,
        "ratio_bootstrap_ci95": ratio_ci,
        "pct_positive_improvement": pct_positive,
        "bootstrap_replicates": 2000,
    }


def _pct_positive_improvement(perf: pd.DataFrame) -> dict[str, float | None]:
    """Share of athlete-seasons with positive improvement (faster later)."""
    imp = _improvement_frame(perf)
    out: dict[str, float | None] = {}
    for gender, label in [("M", "men"), ("F", "women")]:
        vals = imp.loc[imp["gender"] == gender, "improvement_sec"]
        out[label] = round(100.0 * (vals > 0).mean(), 1) if len(vals) else None
    return out
