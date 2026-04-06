"""
Synthetic Theft Injection — Patent Claim 1, Claim 2, Claim 3
Injects gradual fuel drops ONLY during low-speed AND low-RPM windows.
Adds Gaussian noise to simulate real-world sensor uncertainty.
"""

import numpy as np
import pandas as pd

# Physical constraints for theft window selection (Patent Claim 2)
THEFT_SPEED_THRESHOLD = 10.0    # km/h  — vehicle nearly stationary
THEFT_RPM_THRESHOLD   = 1200.0  # RPM   — engine at idle/off

THEFT_DROP_RATE  = 0.5          # % fuel dropped per timestep during theft
THEFT_DURATION   = (20, 60)     # timesteps — min/max theft window length
THEFT_NOISE_STD  = 0.08         # Gaussian noise std (Patent Claim 3)

THEFT_RATIO      = 0.10         # ~10% of eligible windows become theft events


def inject_theft(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Modifies fuel_level in-place for selected low-speed/low-RPM windows.
    Adds 'label' column: 1 = theft, 0 = normal.

    Steps (per vehicle):
      1. Find all time windows where speed < THEFT_SPEED_THRESHOLD
         AND rpm < THEFT_RPM_THRESHOLD (Patent Claim 2 — physical constraint)
      2. Randomly select THEFT_RATIO fraction of those windows
      3. Apply gradual fuel drop + Gaussian noise within selected windows (Claim 3)
      4. Mark those timesteps as label=1
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["label"] = 0

    result_frames = []

    for vid, grp in df.groupby("veh_trip", sort=False):
        grp = grp.copy().reset_index(drop=True)

        # Eligible rows: low speed AND low RPM (Patent Claim 2)
        eligible_mask = (
            (grp["speed_kmh"] < THEFT_SPEED_THRESHOLD) &
            (grp["rpm"]       < THEFT_RPM_THRESHOLD)
        )
        eligible_idx = grp.index[eligible_mask].tolist()

        if len(eligible_idx) < 30:
            result_frames.append(grp)
            continue

        # Select theft start positions
        n_thefts = max(1, int(len(eligible_idx) * THEFT_RATIO / THEFT_DURATION[0]))
        theft_starts = rng.choice(eligible_idx, size=min(n_thefts, len(eligible_idx)), replace=False)

        for start in theft_starts:
            duration = rng.integers(THEFT_DURATION[0], THEFT_DURATION[1])
            end = min(start + duration, len(grp) - 1)
            window_idx = range(start, end + 1)

            # Check all window rows still eligible
            if not eligible_mask.iloc[list(window_idx)].all():
                continue

            # Apply gradual drop + noise (Patent Claim 3)
            for step, i in enumerate(window_idx):
                drop = THEFT_DROP_RATE + rng.normal(0, THEFT_NOISE_STD)
                grp.at[i, "fuel_level"] = max(0.0, grp.at[i, "fuel_level"] - drop)
                grp.at[i, "label"] = 1

        result_frames.append(grp)

    out = pd.concat(result_frames, ignore_index=True)
    print(f"Theft injection complete | Total theft samples: {out['label'].sum():,} / {len(out):,}")
    return out
