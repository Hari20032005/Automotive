"""
Real-time Detector with Consecutive Anomaly Alert Logic — Patent Claim 1 (alert module)
Triggers THEFT ALERT only after N consecutive anomalous windows.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque

from .feature_engineering import FEATURE_COLS
from .lstm_model import WINDOW_SIZE

# --- Alert Logic Parameters (Patent Claim 1) ---
ALERT_THRESHOLD      = 0.5   # probability threshold for a window to be "anomalous"
CONSECUTIVE_N        = 3     # N consecutive positives required to fire alert


class FuelGuardDetector:
    """
    Stateful streaming detector.
    Feed one timestep at a time via .step(); alerts fire after N consecutive hits.
    """

    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self._window: deque = deque(maxlen=WINDOW_SIZE)
        self._consecutive: int = 0
        self.alerts: list[dict] = []

    def step(self, features: dict, timestamp=None) -> dict:
        """
        Process one timestep.

        Parameters
        ----------
        features  : dict with keys matching FEATURE_COLS
        timestamp : optional timestamp label

        Returns
        -------
        dict with keys: prob, anomaly, alert_fired
        """
        row = np.array([features[c] for c in FEATURE_COLS], dtype=np.float32)
        self._window.append(row)

        result = {"prob": None, "anomaly": False, "alert_fired": False, "timestamp": timestamp}

        if len(self._window) < WINDOW_SIZE:
            return result  # not enough data yet

        window_arr = np.array(self._window)[np.newaxis, ...]  # (1, WINDOW_SIZE, n_features)
        prob = float(self.model.predict(window_arr, verbose=0)[0][0])
        anomaly = prob >= ALERT_THRESHOLD
        result["prob"] = prob
        result["anomaly"] = anomaly

        # Consecutive anomaly counter (Patent Claim 1 — alert module)
        if anomaly:
            self._consecutive += 1
        else:
            self._consecutive = 0

        if self._consecutive >= CONSECUTIVE_N:
            result["alert_fired"] = True
            self.alerts.append({"timestamp": timestamp, "prob": prob})
            self._consecutive = 0  # reset after firing

        return result

    def reset(self):
        self._window.clear()
        self._consecutive = 0


def batch_detect(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """
    Run detection on a full vehicle trip DataFrame using vectorised batch prediction.
    Builds ALL windows at once per trip and calls model.predict() once — fast.
    """
    model = tf.keras.models.load_model(model_path)
    df = df.copy()
    df["theft_prob"]      = 0.0
    df["predicted_label"] = 0
    df["alert_fired"]     = False

    for vid, grp in df.groupby("veh_trip", sort=False):
        idx = grp.index.tolist()
        features = grp[FEATURE_COLS].values.astype(np.float32)
        n = len(features)

        if n < WINDOW_SIZE:
            continue

        # Build all windows at once using stride trick (no Python loop)
        n_windows = n - WINDOW_SIZE + 1
        windows = np.stack([features[i : i + WINDOW_SIZE] for i in range(n_windows)])
        # Shape: (n_windows, WINDOW_SIZE, n_features)

        # Single batch prediction call
        raw_probs = model.predict(windows, batch_size=512, verbose=0).flatten()

        # Map window probabilities back to per-row (window i covers row i+WINDOW_SIZE-1)
        probs = np.zeros(n)
        probs[WINDOW_SIZE - 1:] = raw_probs

        predicted = (probs >= ALERT_THRESHOLD).astype(int)

        # Consecutive filter (vectorised)
        alerts = np.zeros(n, dtype=bool)
        consecutive = 0
        for i in range(n):
            if predicted[i] == 1:
                consecutive += 1
                if consecutive >= CONSECUTIVE_N:
                    alerts[i] = True
                    consecutive = 0
            else:
                consecutive = 0

        df.loc[idx, "theft_prob"]      = probs
        df.loc[idx, "predicted_label"] = predicted
        df.loc[idx, "alert_fired"]     = alerts

    return df
