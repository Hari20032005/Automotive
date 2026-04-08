"""
Synthetic Theft Injection — Patent Claim 1, Claim 2, Claim 3
Two theft regimes:
  1. Moving/Idle  — speed < 10 km/h AND RPM < 1200 (vehicle nearly stopped)
  2. Parked       — speed == 0 AND RPM == 0 (engine off, vehicle unattended)
"""

import numpy as np
import pandas as pd

# ── Moving/Idle theft (Patent Claim 2) ───────────────────────────────────────
THEFT_SPEED_THRESHOLD = 10.0    # km/h
THEFT_RPM_THRESHOLD   = 1200.0  # RPM
THEFT_DROP_RATE       = 0.5     # % fuel per timestep
THEFT_DURATION        = (20, 60)
THEFT_NOISE_STD       = 0.08
THEFT_RATIO           = 0.10

# ── Parked theft (engine off, vehicle unattended — siphoning) ─────────────────
PARKED_SPEED_THRESHOLD = 2.0    # km/h — allow tiny GPS noise near zero
PARKED_RPM_THRESHOLD   = 50.0   # RPM  — engine effectively off
PARKED_DROP_RATE       = 1.5    # 3× faster — siphoning is fast
PARKED_DURATION        = (30, 80)
PARKED_THEFT_RATIO     = 0.05


def inject_theft(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Injects two kinds of theft events per vehicle trip:
      Pass 1 — Moving/idle theft (existing regime)
      Pass 2 — Parked theft (new: engine off, faster drain)

    Adds columns:
      label      : 1 = theft, 0 = normal
      theft_type : "moving" | "parked" | "none"
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["label"]      = 0
    df["theft_type"] = "none"

    result_frames = []

    for vid, grp in df.groupby("veh_trip", sort=False):
        grp = grp.copy().reset_index(drop=True)

        # ── Pass 1: Moving / Idle theft ───────────────────────────────────
        moving_mask = (
            (grp["speed_kmh"] < THEFT_SPEED_THRESHOLD) &
            (grp["rpm"]       < THEFT_RPM_THRESHOLD)
        )
        moving_idx = grp.index[moving_mask].tolist()

        if len(moving_idx) >= 30:
            n_thefts = max(1, int(len(moving_idx) * THEFT_RATIO / THEFT_DURATION[0]))
            starts   = rng.choice(moving_idx, size=min(n_thefts, len(moving_idx)), replace=False)

            for start in starts:
                duration  = rng.integers(THEFT_DURATION[0], THEFT_DURATION[1])
                end       = min(start + duration, len(grp) - 1)
                window    = range(start, end + 1)
                if not moving_mask.iloc[list(window)].all():
                    continue
                for i in window:
                    drop = THEFT_DROP_RATE + rng.normal(0, THEFT_NOISE_STD)
                    grp.at[i, "fuel_level"]  = max(0.0, grp.at[i, "fuel_level"] - drop)
                    grp.at[i, "label"]       = 1
                    grp.at[i, "theft_type"]  = "moving"

        # ── Pass 2: Parked theft (engine off) ─────────────────────────────
        parked_mask = (
            (grp["speed_kmh"] <= PARKED_SPEED_THRESHOLD) &
            (grp["rpm"]       <= PARKED_RPM_THRESHOLD) &
            (grp["label"]     == 0)           # don't double-label
        )
        parked_idx = grp.index[parked_mask].tolist()

        if len(parked_idx) >= 20:
            n_thefts = max(1, int(len(parked_idx) * PARKED_THEFT_RATIO / PARKED_DURATION[0]))
            starts   = rng.choice(parked_idx, size=min(n_thefts, len(parked_idx)), replace=False)

            for start in starts:
                duration = rng.integers(PARKED_DURATION[0], PARKED_DURATION[1])
                end      = min(start + duration, len(grp) - 1)
                window   = range(start, end + 1)
                # Ensure entire window is still parked & unlabeled
                if not parked_mask.iloc[list(window)].all():
                    continue
                for i in window:
                    drop = PARKED_DROP_RATE + rng.normal(0, THEFT_NOISE_STD)
                    grp.at[i, "fuel_level"]  = max(0.0, grp.at[i, "fuel_level"] - drop)
                    grp.at[i, "label"]       = 1
                    grp.at[i, "theft_type"]  = "parked"

        result_frames.append(grp)

    out = pd.concat(result_frames, ignore_index=True)
    moving_n = (out["theft_type"] == "moving").sum()
    parked_n = (out["theft_type"] == "parked").sum()
    print(f"Theft injection | moving: {moving_n:,} | parked: {parked_n:,} | total: {out['label'].sum():,} / {len(out):,}")
    return out
