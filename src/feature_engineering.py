"""
Feature Engineering — Patent Claim 1 & Claim 7
Computes: fuel_delta, rolling stats, speed-RPM interaction, per-vehicle normalization.
"""

import numpy as np
import pandas as pd


ROLLING_WINDOW = 10   # timesteps for rolling stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to the DataFrame per vehicle:

    Features (Patent Claims 1, 5, 7):
      fuel_delta          — rate of change of fuel level between consecutive samples
      fuel_roll_mean      — rolling mean of fuel level (window=ROLLING_WINDOW)
      fuel_roll_std       — rolling std of fuel level
      speed_roll_mean     — rolling mean of speed
      rpm_roll_mean       — rolling mean of RPM
      speed_rpm_interact  — normalized_speed × normalized_rpm (interaction term)
      fuel_norm           — per-vehicle z-score normalized fuel level
      speed_norm          — per-vehicle z-score normalized speed
      rpm_norm            — per-vehicle z-score normalized rpm
    """
    df = df.copy()
    feature_cols = []

    result_frames = []

    for vid, grp in df.groupby("veh_trip", sort=False):
        grp = grp.copy().reset_index(drop=True)

        # --- Fuel delta ---
        grp["fuel_delta"] = grp["fuel_level"].diff().fillna(0)

        # --- Rolling stats ---
        grp["fuel_roll_mean"] = grp["fuel_level"].rolling(ROLLING_WINDOW, min_periods=1).mean()
        grp["fuel_roll_std"]  = grp["fuel_level"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
        grp["speed_roll_mean"] = grp["speed_kmh"].rolling(ROLLING_WINDOW, min_periods=1).mean()
        grp["rpm_roll_mean"]   = grp["rpm"].rolling(ROLLING_WINDOW, min_periods=1).mean()

        # --- Per-vehicle normalization (z-score) — Patent Claim 7 ---
        for col, norm_col in [
            ("fuel_level", "fuel_norm"),
            ("speed_kmh",  "speed_norm"),
            ("rpm",        "rpm_norm"),
        ]:
            mu  = grp[col].mean()
            std = grp[col].std() + 1e-8
            grp[norm_col] = (grp[col] - mu) / std

        # --- Speed-RPM interaction term — Patent Claim 5 ---
        grp["speed_rpm_interact"] = grp["speed_norm"] * grp["rpm_norm"]

        result_frames.append(grp)

    out = pd.concat(result_frames, ignore_index=True)
    return out


FEATURE_COLS = [
    "fuel_level",
    "fuel_delta",
    "fuel_roll_mean",
    "fuel_roll_std",
    "speed_kmh",
    "speed_roll_mean",
    "rpm",
    "rpm_roll_mean",
    "speed_rpm_interact",
    "fuel_norm",
    "speed_norm",
    "rpm_norm",
]
