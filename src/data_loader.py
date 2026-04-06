"""
VED (Vehicle Energy Dataset) Data Loader
Dataset: /VED-master/Data/extracted/VED_*.csv

Real column names in VED CSVs:
  VehId, Trip, Timestamp(ms), Vehicle Speed[km/h], Engine RPM[RPM], Fuel Rate[L/hr]

Only ICE (gasoline) vehicles have Fuel Rate data — we filter for those.
Grouping key: (VehId, Trip) — each trip is a separate drive session.
"""

import os
import glob
import numpy as np
import pandas as pd

VED_COL_MAP = {
    "VehId":                  "vehicle_id",
    "Trip":                   "trip_id",
    "Timestamp(ms)":          "timestamp_ms",
    "Vehicle Speed[km/h]":    "speed_kmh",
    "Engine RPM[RPM]":        "rpm",
    "Fuel Rate[L/hr]":        "fuel_rate",
}

USECOLS = list(VED_COL_MAP.keys())

# Minimum rows per trip to be considered usable
MIN_TRIP_ROWS = 50


def load_ved_data(
    data_dir: str,
    max_vehicles: int | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Load all VED_*.csv files from data_dir.
    Keeps only ICE vehicles (rows where Fuel Rate is present).
    Groups by (VehId, Trip).

    Parameters
    ----------
    data_dir      : folder containing VED_*.csv files
    max_vehicles  : limit to first N unique vehicle IDs after loading
    max_files     : limit number of CSV files to read (faster for testing)

    Returns
    -------
    pd.DataFrame with columns:
        vehicle_id, trip_id, timestamp_ms, speed_kmh, rpm, fuel_rate
        + composite 'veh_trip' key
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            "Point data_dir to the extracted VED folder:\n"
            "  .../VED-master/Data/extracted/"
        )

    if max_files:
        csv_files = csv_files[:max_files]

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
        # Keep only ICE rows: fuel rate, speed, RPM must all be present
        mask = (
            df["Fuel Rate[L/hr]"].notna() &
            df["Vehicle Speed[km/h]"].notna() &
            df["Engine RPM[RPM]"].notna() &
            (df["Fuel Rate[L/hr]"] >= 0)
        )
        df = df[mask]
        if len(df):
            frames.append(df)
        print(f"  {os.path.basename(path)}: {mask.sum():,} ICE rows")

    if not frames:
        raise ValueError("No ICE vehicle rows found. Check that Fuel Rate[L/hr] is present.")

    raw = pd.concat(frames, ignore_index=True)
    raw.rename(columns=VED_COL_MAP, inplace=True)

    # Composite key: vehicle + trip
    raw["veh_trip"] = raw["vehicle_id"].astype(str) + "_T" + raw["trip_id"].astype(str)

    # Sort by vehicle, trip, time
    raw.sort_values(["vehicle_id", "trip_id", "timestamp_ms"], inplace=True)
    raw.reset_index(drop=True, inplace=True)

    # Drop very short trips
    trip_sizes = raw.groupby("veh_trip").size()
    valid_trips = trip_sizes[trip_sizes >= MIN_TRIP_ROWS].index
    raw = raw[raw["veh_trip"].isin(valid_trips)]

    if max_vehicles:
        vehicle_ids = raw["vehicle_id"].unique()[:max_vehicles]
        raw = raw[raw["vehicle_id"].isin(vehicle_ids)]

    raw.reset_index(drop=True, inplace=True)

    print(f"\nLoaded {len(raw):,} rows | {raw['vehicle_id'].nunique()} vehicles | {raw['veh_trip'].nunique()} trips")
    return raw


def reconstruct_fuel_level(df: pd.DataFrame, tank_litres: float = 50.0) -> pd.DataFrame:
    """
    VED records fuel RATE (L/hr) not fuel LEVEL.
    Reconstruct cumulative fuel consumed per trip, convert to 0-100% level.

    Assumes tank starts at 100% at the beginning of each trip.
    """
    df = df.copy()
    df["fuel_level"] = np.nan

    for vt, grp in df.groupby("veh_trip", sort=False):
        idx = grp.index

        # Time delta in hours between samples
        dt_ms = grp["timestamp_ms"].diff().fillna(0).clip(lower=0)
        dt_hr = dt_ms / 3_600_000.0

        fuel_consumed = (grp["fuel_rate"] * dt_hr).cumsum()
        level = 100.0 - (fuel_consumed / tank_litres * 100.0)
        level = level.clip(0.0, 100.0)

        df.loc[idx, "fuel_level"] = level.values

    return df
