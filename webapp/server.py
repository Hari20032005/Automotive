"""
FuelGuard AI — Flask Backend
Run: python3 webapp/server.py
Open: http://localhost:5050
"""

import sys, os, threading, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, jsonify, request

from src.data_loader import load_ved_data, reconstruct_fuel_level
from src.feature_engineering import engineer_features, FEATURE_COLS
from src.theft_injector import inject_theft, PARKED_DROP_RATE, THEFT_NOISE_STD
from src.detector import batch_detect, ALERT_THRESHOLD, CONSECUTIVE_N
from src.lstm_model import WINDOW_SIZE

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
VED_DIR    = str(BASE_DIR.parent / "VED-master" / "Data" / "extracted")
MODEL_PATH = str(BASE_DIR / "fuelguard_lstm.keras")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")

# ── Data cache ────────────────────────────────────────────────────────────────
_cache      = {}
_cache_lock = threading.Lock()
_loading    = threading.Event()


def _load_data():
    """Load full pipeline in background on startup."""
    print("[FuelGuard] Loading VED data...")
    df = load_ved_data(VED_DIR, max_files=3)
    df = reconstruct_fuel_level(df)
    df = engineer_features(df)
    df = inject_theft(df, seed=42)
    df = batch_detect(df, model_path=MODEL_PATH)
    with _cache_lock:
        _cache["df"] = df
    _loading.set()
    print("[FuelGuard] Data ready.")


def get_df() -> pd.DataFrame:
    _loading.wait()
    return _cache["df"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def df_to_rows(grp: pd.DataFrame) -> list:
    cols = ["fuel_level", "speed_kmh", "rpm", "theft_prob",
            "label", "theft_type", "alert_fired"]
    records = []
    for local_i, (_, row) in enumerate(grp.iterrows()):
        records.append({
            "i":          local_i,
            "fuel_level": round(float(row["fuel_level"]), 3),
            "speed_kmh":  round(float(row["speed_kmh"]), 2),
            "rpm":        round(float(row["rpm"]), 1),
            "theft_prob": round(float(row["theft_prob"]), 4),
            "label":      int(row["label"]),
            "theft_type": str(row.get("theft_type", "none")),
            "alert_fired":bool(row["alert_fired"]),
        })
    return records


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    ready = _loading.is_set()
    return jsonify({"ready": ready})


@app.route("/api/stats")
def stats():
    if not _loading.is_set():
        return jsonify({"ready": False}), 202
    df = get_df()
    return jsonify({
        "ready":          True,
        "total_vehicles": int(df["vehicle_id"].nunique()),
        "total_trips":    int(df["veh_trip"].nunique()),
        "total_alerts":   int(df["alert_fired"].sum()),
        "moving_thefts":  int((df["theft_type"] == "moving").sum()),
        "parked_thefts":  int((df["theft_type"] == "parked").sum()),
        "model_auc":      0.9605,
        "model_recall":   1.0,
    })


@app.route("/api/trips")
def trips():
    if not _loading.is_set():
        return jsonify({"ready": False}), 202
    df = get_df()
    result = []
    for vt, grp in df.groupby("veh_trip", sort=False):
        has_parked   = bool((grp["theft_type"] == "parked").any())
        has_moving   = bool((grp["theft_type"] == "moving").any())
        has_theft    = has_parked or has_moving
        n_alerts     = int(grp["alert_fired"].sum())
        theft_types  = []
        if has_parked: theft_types.append("parked")
        if has_moving: theft_types.append("moving")
        # COMPROMISED = actual theft injected in this trip
        status = "COMPROMISED" if has_theft else "SECURE"
        result.append({
            "veh_trip":    vt,
            "vehicle_id":  str(grp["vehicle_id"].iloc[0]),
            "n_samples":   len(grp),
            "has_theft":   has_theft,
            "theft_types": theft_types,
            "n_alerts":    n_alerts,
            "status":      status,
            "fuel_start":  round(float(grp["fuel_level"].iloc[0]), 1),
            "fuel_end":    round(float(grp["fuel_level"].iloc[-1]), 1),
            "max_prob":    round(float(grp["theft_prob"].max()), 3),
        })
    # Sort: COMPROMISED first, then by theft types (parked first)
    result.sort(key=lambda x: (
        0 if x["status"] == "COMPROMISED" else 1,
        0 if "parked" in x["theft_types"] else 1,
        x["veh_trip"]
    ))
    return jsonify(result)


@app.route("/api/trip/<veh_trip>")
def trip(veh_trip):
    if not _loading.is_set():
        return jsonify({"ready": False}), 202
    df  = get_df()
    grp = df[df["veh_trip"] == veh_trip]
    if len(grp) == 0:
        return jsonify({"error": "Trip not found"}), 404
    return jsonify({"veh_trip": veh_trip, "rows": df_to_rows(grp)})


@app.route("/api/simulate-theft", methods=["POST"])
def simulate_theft():
    """
    Injects a PARKED theft event at the given position in a trip.
    Returns only the modified rows (injection window).
    Does NOT mutate the cache.
    """
    if not _loading.is_set():
        return jsonify({"ready": False}), 202

    body     = request.get_json()
    veh_trip = body.get("veh_trip")
    position = int(body.get("position", 0))
    duration = int(body.get("duration", 40))

    df  = get_df()
    grp = df[df["veh_trip"] == veh_trip].reset_index(drop=True)
    if len(grp) == 0:
        return jsonify({"error": "Trip not found"}), 404

    # Context window for accurate rolling stats
    ctx_start = max(0, position - 60)
    ctx_end   = min(len(grp) - 1, position + duration + 60)
    ctx       = grp.iloc[ctx_start : ctx_end + 1].copy().reset_index(drop=True)

    # Injection offset within context
    inj_start = position - ctx_start
    inj_end   = min(inj_start + duration, len(ctx) - 1)

    rng = np.random.default_rng(99)

    # Apply parked theft: zero speed/rpm, drain fuel
    for i in range(inj_start, inj_end + 1):
        ctx.at[i, "speed_kmh"]  = 0.0
        ctx.at[i, "rpm"]        = 0.0
        drop = PARKED_DROP_RATE + rng.normal(0, THEFT_NOISE_STD)
        ctx.at[i, "fuel_level"] = max(0.0, ctx.at[i, "fuel_level"] - drop)
        ctx.at[i, "label"]      = 1
        ctx.at[i, "theft_type"] = "parked"

    # Re-engineer features on context
    ctx_feat = engineer_features(ctx.assign(veh_trip=veh_trip))

    # Re-run detection on context
    ctx_feat["vehicle_id"] = grp["vehicle_id"].iloc[0]
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)

    features = ctx_feat[FEATURE_COLS].values.astype("float32")
    n = len(features)
    probs = np.zeros(n)
    if n >= WINDOW_SIZE:
        n_windows = n - WINDOW_SIZE + 1
        windows   = np.stack([features[i: i + WINDOW_SIZE] for i in range(n_windows)])
        raw       = model.predict(windows, batch_size=256, verbose=0).flatten()
        probs[WINDOW_SIZE - 1:] = raw

    predicted = (probs >= ALERT_THRESHOLD).astype(int)
    alerts    = np.zeros(n, dtype=bool)
    consec    = 0
    for i in range(n):
        if predicted[i]:
            consec += 1
            if consec >= CONSECUTIVE_N:
                alerts[i] = True
                consec = 0
        else:
            consec = 0

    ctx_feat["theft_prob"]  = probs
    ctx_feat["alert_fired"] = alerts

    # Return only the injection window (mapped back to original positions)
    result_rows = []
    for local_i in range(inj_start, inj_end + 1):
        orig_i = ctx_start + local_i
        row    = ctx_feat.iloc[local_i]
        result_rows.append({
            "i":          orig_i,
            "fuel_level": round(float(row["fuel_level"]), 3),
            "speed_kmh":  round(float(row["speed_kmh"]), 2),
            "rpm":        round(float(row["rpm"]), 1),
            "theft_prob": round(float(row["theft_prob"]), 4),
            "label":      1,
            "theft_type": "parked",
            "alert_fired":bool(alerts[local_i]),
        })

    return jsonify({"injected": result_rows, "position": position, "duration": duration})


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-load data in background thread
    t = threading.Thread(target=_load_data, daemon=True)
    t.start()
    print("[FuelGuard] Starting server at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)
