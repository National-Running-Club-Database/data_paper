"""Gender-stratified R²: converted-only vs full standardization (train 2023, test 2024).

Illustrative checks cited in the CIKM paper (RF + gradient boosting on season features).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from load_data import load_tables
from progress import iter_progress
from standardization import compute_xc_times
from xc_frame import build_xc_frame

FEATURE_COLUMNS = [
    "gender_encoded",
    "year",
    "num_races",
    "season_duration",
    "first_time",
    "last_time",
    "best_time",
    "worst_time",
    "avg_time",
    "time_std",
    "time_range",
    "cv_time",
    "race_frequency",
    "starting_percentile",
    "gender_year",
    "starting_percentile_squared",
    "num_races_squared",
    "season_duration_squared",
    "best_to_avg_ratio",
    "worst_to_avg_ratio",
    "variability_score",
    "consistency_score",
    "experience_level",
    "slope",
    "avg_days_between_races",
    "race_to_race_improvement_std",
    "best_race_timing",
    "best_race_timing_ratio",
    "bad_race_count",
]

MODEL_BUILDERS: dict[str, Callable[[], object]] = {
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=42),
}


def _xc_race_frames(
    tables: dict, *, progress_disable: bool = False
) -> dict[str, pd.DataFrame]:
    """Build converted and standardized XC race frames in one standardization pass."""
    base = build_xc_frame(tables, exclude_nationals=True)
    base = base[base["event_category"] == "distance"].copy()
    timed = compute_xc_times(base)
    cols = ["athlete_id", "gender", "race_date"]

    frames = {}
    for mode, col in (("converted", "converted_sec"), ("standardized", "standardized_sec")):
        out = timed.loc[np.isfinite(timed[col]), cols + [col]].copy()
        out = out.rename(columns={col: "standardized_to_target", "race_date": "start_date"})
        frames[mode] = out.dropna(subset=["start_date", "gender"])
    return frames


def _athlete_features(df: pd.DataFrame, training_df: pd.DataFrame | None = None) -> pd.DataFrame:
    percentile_df = training_df if training_df is not None else df
    records = []

    for athlete_id, athlete_races in df.groupby("athlete_id"):
        athlete_races = athlete_races.sort_values("start_date")
        if len(athlete_races) < 2:
            continue

        first_time = athlete_races.iloc[0]["standardized_to_target"]
        last_time = athlete_races.iloc[-1]["standardized_to_target"]
        first_date = athlete_races.iloc[0]["start_date"]
        last_date = athlete_races.iloc[-1]["start_date"]
        days_diff = (last_date - first_date).days
        if days_diff < 7:
            continue

        total_improvement = last_time - first_time
        improvement_rate = total_improvement / days_diff
        num_races = len(athlete_races)
        season_duration = days_diff
        times = athlete_races["standardized_to_target"].values
        best_time = float(np.min(times))
        worst_time = float(np.max(times))
        avg_time = float(np.mean(times))
        time_std = float(np.std(times))
        time_range = worst_time - best_time
        cv_time = time_std / avg_time if avg_time > 0 else 0.0

        if len(times) >= 3:
            xs = np.arange(len(times) - 1).reshape(-1, 1)
            ys = times[:-1]
            slope = float(LinearRegression().fit(xs, ys).coef_[0])
        elif len(times) == 2:
            slope = float(times[1] - times[0])
        else:
            slope = 0.0

        race_frequency = num_races / season_duration if season_duration > 0 else 0.0
        if num_races > 1:
            gaps = athlete_races["start_date"].diff().dropna()
            avg_days_between_races = float(gaps.dt.days.mean())
        else:
            avg_days_between_races = 0.0

        if len(times) >= 2:
            diffs = np.diff(times)
            race_to_race_improvement_std = float(np.std(diffs))
            bad_race_count = int(np.sum(diffs > 0))
        else:
            race_to_race_improvement_std = 0.0
            bad_race_count = 0

        best_idx = int(np.argmin(times))
        if best_idx == 0:
            best_race_timing = 0.0
        elif best_idx == len(times) - 1:
            best_race_timing = float(season_duration)
        else:
            best_race_timing = float(
                (athlete_races.iloc[best_idx]["start_date"] - first_date).days
            )

        gender = athlete_races.iloc[0]["gender"]
        year = int(first_date.year)
        pg = percentile_df[
            (percentile_df["start_date"].dt.year < year)
            & (percentile_df["gender"] == gender)
        ]
        if len(pg) == 0:
            pg = percentile_df[percentile_df["gender"] == gender]
        starting_percentile = (
            float((pg["standardized_to_target"] <= first_time).mean() * 100) if len(pg) else 50.0
        )

        records.append(
            {
                "athlete_id": athlete_id,
                "gender": gender,
                "year": year,
                "num_races": num_races,
                "season_duration": season_duration,
                "first_time": first_time,
                "last_time": last_time,
                "best_time": best_time,
                "worst_time": worst_time,
                "avg_time": avg_time,
                "time_std": time_std,
                "time_range": time_range,
                "cv_time": cv_time,
                "total_improvement": total_improvement,
                "improvement_rate": improvement_rate,
                "slope": slope,
                "race_frequency": race_frequency,
                "starting_percentile": starting_percentile,
                "avg_days_between_races": avg_days_between_races,
                "race_to_race_improvement_std": race_to_race_improvement_std,
                "best_race_timing": best_race_timing,
                "bad_race_count": bad_race_count,
            }
        )

    return pd.DataFrame(records)


def _advanced_features(athlete_df: pd.DataFrame) -> pd.DataFrame:
    f = athlete_df.copy()
    le = LabelEncoder()
    f["gender_encoded"] = le.fit_transform(f["gender"])
    f["gender_year"] = f["gender_encoded"] * f["year"]
    f["starting_percentile_squared"] = f["starting_percentile"] ** 2
    f["num_races_squared"] = f["num_races"] ** 2
    f["season_duration_squared"] = f["season_duration"] ** 2
    f["best_to_avg_ratio"] = f["best_time"] / f["avg_time"]
    f["worst_to_avg_ratio"] = f["worst_time"] / f["avg_time"]
    f["variability_score"] = np.where(
        f["time_range"] > 0,
        1 / (1 + f["time_range"] / f["avg_time"]),
        1.0,
    )
    f["consistency_score"] = 1 / (1 + f["cv_time"])
    f["experience_level"] = f["num_races"] * f["season_duration"]
    f["best_race_timing_ratio"] = (
        f["best_race_timing"] / f["season_duration"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0)
    return f


def _prepare_xy(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = features_df[FEATURE_COLUMNS].copy()
    y = features_df["improvement_rate"].copy()
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    mask &= (y >= -50) & (y <= 50)
    filt = features_df.loc[mask].copy()
    return X.loc[mask], y.loc[mask], filt


def _fit_predict_r2(
    model_builder: Callable[[], object],
    race_frames: dict[str, pd.DataFrame],
    mode: str,
    gender: str,
) -> dict:
    df = race_frames[mode].copy()
    df["year"] = df["start_date"].dt.year
    training = df[df["year"] == 2023]
    athlete_df = _athlete_features(df, training_df=training)
    features_df = _advanced_features(athlete_df)
    X, y, filt = _prepare_xy(features_df)
    gmask = filt["gender"] == gender
    train = gmask & (filt["year"] == 2023)
    test = gmask & (filt["year"] == 2024)
    if train.sum() < 50 or test.sum() < 50:
        return {"r2": None, "train_n": int(train.sum()), "test_n": int(test.sum())}
    model = model_builder()
    model.fit(X.loc[train], y.loc[train])
    pred = model.predict(X.loc[test])
    return {
        "r2": round(float(r2_score(y.loc[test], pred)), 3),
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
    }


def _bootstrap_r2_diff_ci(
    model_builder: Callable[[], object],
    race_frames: dict[str, pd.DataFrame],
    gender: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap 95% CIs for R² and converted−standardized difference on test set."""
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mode in ("converted", "standardized"):
        df = race_frames[mode].copy()
        df["year"] = df["start_date"].dt.year
        training = df[df["year"] == 2023]
        athlete_df = _athlete_features(df, training_df=training)
        features_df = _advanced_features(athlete_df)
        X, y, filt = _prepare_xy(features_df)
        gmask = filt["gender"] == gender
        train = gmask & (filt["year"] == 2023)
        test = gmask & (filt["year"] == 2024)
        if train.sum() < 50 or test.sum() < 50:
            return {"bootstrap_replicates": n_boot}
        model = model_builder()
        model.fit(X.loc[train], y.loc[train])
        preds[mode] = (y.loc[test].values, model.predict(X.loc[test]))

    y_c, p_c = preds["converted"]
    y_s, p_s = preds["standardized"]
    rng = np.random.default_rng(seed)
    n = len(y_c)
    r2_c_vals, r2_s_vals, diffs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r2_c = r2_score(y_c[idx], p_c[idx])
        r2_s = r2_score(y_s[idx], p_s[idx])
        r2_c_vals.append(r2_c)
        r2_s_vals.append(r2_s)
        diffs.append(r2_c - r2_s)

    def ci(vals: list[float]) -> dict[str, float]:
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"lo": round(float(lo), 3), "hi": round(float(hi), 3)}

    return {
        "bootstrap_replicates": n_boot,
        "converted_only_r2_ci95": ci(r2_c_vals),
        "standardized_r2_ci95": ci(r2_s_vals),
        "r2_diff_conv_minus_std_ci95": ci(diffs),
    }


def r2_for_mode_gender(
    race_frames: dict[str, pd.DataFrame],
    mode: str,
    gender: str,
    *,
    model_name: str = "random_forest",
) -> dict:
    return _fit_predict_r2(MODEL_BUILDERS[model_name], race_frames, mode, gender)


def main(*, progress_disable: bool = False, bootstrap: bool = True) -> None:
    tables = load_tables()
    race_frames = _xc_race_frames(tables, progress_disable=progress_disable)
    out: dict = {}
    tasks = [
        (model_name, gender_code, gender_label, mode)
        for model_name in MODEL_BUILDERS
        for gender_code, gender_label in (("M", "men"), ("F", "women"))
        for mode in ("converted", "standardized")
    ]
    for model_name, gender_code, gender_label, mode in iter_progress(
        tasks, desc="ML R²", unit="model", disable=progress_disable
    ):
        out.setdefault(model_name, {}).setdefault(gender_label, {})
        key = f"{mode}_only" if mode == "converted" else mode
        out[model_name][gender_label][key] = r2_for_mode_gender(
            race_frames, mode, gender_code, model_name=model_name
        )

    if bootstrap:
        for model_name in MODEL_BUILDERS:
            for gender_code, gender_label in (("M", "men"), ("F", "women")):
                out[model_name][gender_label]["bootstrap"] = _bootstrap_r2_diff_ci(
                    MODEL_BUILDERS[model_name], race_frames, gender_code
                )

    # Legacy flat layout (random forest only) for older consumers.
    out["men"] = out["random_forest"]["men"]
    out["women"] = out["random_forest"]["women"]

    path = ROOT / "results" / "ml_standardization_r2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
