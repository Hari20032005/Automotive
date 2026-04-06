# FuelGuard AI — LSTM-Based Fuel Theft Detection

**Team:** FuelGuard AI (Ashissh S, Hariharan S, Harsha)  
**Patent Status:** Novelty confirmed — draft filed (see `../PATENT_DRAFT.md`)

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. VED Dataset (already extracted)
The VED CSVs are at:
```
../VED-master/Data/extracted/VED_*.csv   (54 weekly files, ICE vehicles only)
```
The default `--data_dir` already points to this path automatically.

### 3. Train the model
```bash
# Full training (all 54 files, all ICE vehicles)
python main.py

# Quick test (first 5 files only)
python main.py --max_files 5
```
This runs the full pipeline:
- Loads VED data (fuel rate → reconstructed fuel level, speed, RPM)
- Engineers features (fuel delta, rolling stats, speed×RPM interaction)
- Injects synthetic theft events (low-speed + low-RPM constraint)
- Trains stacked LSTM binary classifier
- Saves model to `fuelguard_lstm.keras`
- Shows confusion matrix + ROC-AUC plot

### 4. Launch the faculty demo dashboard
```bash
streamlit run dashboard/app.py
```

---

## Patent-Mapped Architecture

| Component | File | Patent Claim |
|-----------|------|-------------|
| Multi-signal VED loader | `src/data_loader.py` | Claim 1 (data acquisition) |
| Fuel delta + rolling stats + speed×RPM | `src/feature_engineering.py` | Claims 1, 5, 7 |
| Low-speed + low-RPM theft injection | `src/theft_injector.py` | Claims 2, 3 |
| Stacked LSTM + sigmoid | `src/lstm_model.py` | Claim 4 |
| Consecutive-N alert logic | `src/detector.py` | Claim 1 (alert module) |
| Live dashboard | `dashboard/app.py` | Demo |

---

## VED Dataset Columns Used

| VED Column | Renamed To | Used For |
|-----------|-----------|---------|
| `Vehicle Speed[km/h]` | `speed_kmh` | Context signal, theft window selection |
| `Engine RPM[RPM]` | `rpm` | Context signal, theft window selection |
| `Fuel Rate[L/hr]` | `fuel_rate` | Reconstructed into `fuel_level` |
| `VehId` | `vehicle_id` | Per-vehicle normalization |
| `Timestamp(ms)` | `timestamp_ms` | Time-series ordering |
# Automotive
