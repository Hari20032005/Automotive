"""
FuelGuard AI — Main Training Pipeline
Uses real VED (Vehicle Energy Dataset) data.

Usage:
  python main.py --data_dir data/ --max_vehicles 20
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from src.data_loader import load_ved_data, reconstruct_fuel_level
from src.feature_engineering import engineer_features
from src.theft_injector import inject_theft
from src.lstm_model import train


def parse_args():
    p = argparse.ArgumentParser(description="FuelGuard AI Training Pipeline")
    VED_DEFAULT = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "VED-master", "Data", "extracted"
    )
    p.add_argument("--data_dir",      default=VED_DEFAULT,           help="Folder containing VED CSV files")
    p.add_argument("--max_vehicles",  type=int, default=None,         help="Limit number of vehicles loaded")
    p.add_argument("--max_files",     type=int, default=None,         help="Limit number of CSV files to read (faster testing)")
    p.add_argument("--model_out",     default="fuelguard_lstm.keras", help="Output model file")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def evaluate(model, X_test, y_test, save_dir="."):
    """Print metrics and save plots."""
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Theft"]))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    # --- Confusion Matrix ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Normal", "Theft"],
                yticklabels=["Normal", "Theft"], ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="crimson", lw=2, label=f"ROC AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve — FuelGuard AI")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, "evaluation_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"Evaluation plots saved to: {out_path}")
    plt.show()


def main():
    args = parse_args()

    print("="*50)
    print("FuelGuard AI — LSTM Fuel Theft Detection")
    print("Patent: Multi-Signal LSTM + Synthetic Injection")
    print("="*50)

    # Step 1: Load VED data
    print("\n[1/5] Loading VED data...")
    df = load_ved_data(args.data_dir, max_vehicles=args.max_vehicles, max_files=args.max_files)

    # Step 2: Reconstruct fuel level from fuel rate
    print("\n[2/5] Reconstructing fuel level from fuel rate...")
    df = reconstruct_fuel_level(df)

    # Step 3: Feature engineering
    print("\n[3/5] Engineering features...")
    df = engineer_features(df)

    # Step 4: Synthetic theft injection
    print("\n[4/5] Injecting synthetic theft events...")
    df = inject_theft(df, seed=args.seed)

    # Step 5: Train LSTM
    print("\n[5/5] Training LSTM model...")
    model, history, X_test, y_test = train(df, model_save_path=args.model_out)

    # Evaluate
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
