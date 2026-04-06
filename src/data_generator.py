"""
Synthetic Vehicle Telematics Data Generator
Simulates Vehicle Energy Dataset (VED) style data: fuel level, speed, RPM.
"""

import numpy as np
import pandas as pd


def generate_vehicle_trip(
    vehicle_id: str,
    n_samples: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a realistic vehicle telematics trip time series.

    Returns DataFrame with columns:
      timestamp, vehicle_id, fuel_level, speed_kmh, rpm
    """
    rng = np.random.default_rng(seed)

    # --- Drive cycle simulation ---
    t = np.linspace(0, n_samples / 10, n_samples)  # 10Hz sampling

    # Speed: sinusoidal with noise (0-100 kmh)
    speed_base = 40 + 30 * np.sin(2 * np.pi * t / 120) + 15 * np.sin(2 * np.pi * t / 30)
    speed = np.clip(speed_base + rng.normal(0, 3, n_samples), 0, 120)

    # Idle/stop phases (every ~5 minutes)
    stop_mask = (t % 300) < 20
    speed[stop_mask] = rng.uniform(0, 2, stop_mask.sum())

    # RPM: correlated with speed (idle ~800, moving ~2500)
    rpm_base = 800 + (speed / 120) * 2200
    rpm = np.clip(rpm_base + rng.normal(0, 100, n_samples), 600, 5000)

    # Fuel level: starts at 75%, gradual consumption based on speed/rpm
    consumption_rate = 0.000015 + (speed / 120) * 0.00004 + (rpm / 5000) * 0.00002
    fuel_deltas = -consumption_rate + rng.normal(0, 0.00001, n_samples)  # sensor noise
    fuel_level = 75.0 + np.cumsum(fuel_deltas)
    fuel_level = np.clip(fuel_level, 0, 100)

    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="100ms")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "vehicle_id": vehicle_id,
        "fuel_level": fuel_level,
        "speed_kmh": speed,
        "rpm": rpm,
        "label": 0  # 0 = normal
    })
    return df


def generate_fleet_dataset(
    n_vehicles: int = 5,
    n_samples_per_vehicle: int = 2000,
    seed: int = 0
) -> pd.DataFrame:
    """Generate multi-vehicle dataset."""
    dfs = []
    for i in range(n_vehicles):
        df = generate_vehicle_trip(
            vehicle_id=f"VH-{i+1:03d}",
            n_samples=n_samples_per_vehicle,
            seed=seed + i * 7
        )
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
