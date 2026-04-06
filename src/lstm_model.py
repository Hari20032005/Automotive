"""
LSTM Model — Patent Claim 4
Two stacked LSTM layers + Dense sigmoid output (binary classifier).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from .feature_engineering import FEATURE_COLS

# --- Hyperparameters (Patent Claim 6: window 5-10 timesteps) ---
WINDOW_SIZE  = 10   # timesteps per sequence
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.3
BATCH_SIZE   = 64
EPOCHS       = 30


def build_sequences(df: pd.DataFrame):
    """
    Build sliding-window sequences (X) and labels (y) per vehicle.
    Patent Claim 6: window length = WINDOW_SIZE (5-10 timesteps).
    """
    X_list, y_list = [], []

    for vid, grp in df.groupby("veh_trip", sort=False):
        grp = grp.reset_index(drop=True)
        features = grp[FEATURE_COLS].values.astype(np.float32)
        labels   = grp["label"].values.astype(np.float32)

        for i in range(len(grp) - WINDOW_SIZE):
            X_list.append(features[i : i + WINDOW_SIZE])
            y_list.append(labels[i + WINDOW_SIZE - 1])  # label of last step in window

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def build_model(n_features: int) -> tf.keras.Model:
    """
    Patent Claim 4: Two stacked LSTM layers + Dense sigmoid (binary cross-entropy).
    """
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        LSTM(LSTM_UNITS_1, return_sequences=True, name="lstm_1"),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS_2, return_sequences=False, name="lstm_2"),
        Dropout(DROPOUT_RATE),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid", name="theft_prob"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="roc_auc")]
    )
    return model


def train(df: pd.DataFrame, model_save_path: str = "fuelguard_lstm.keras"):
    """
    Full training pipeline:
      1. Build sequences
      2. Train/val split
      3. Train model with early stopping
      4. Save model
    Returns: model, history, X_test, y_test
    """
    print("Building sliding-window sequences...")
    X, y = build_sequences(df)
    print(f"  X shape: {X.shape} | y shape: {y.shape}")
    print(f"  Theft ratio: {y.mean():.3%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_features = X.shape[2]
    model = build_model(n_features)
    model.summary()

    # Class weighting for imbalanced data
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg / (pos + 1e-8)}
    print(f"Class weights — normal: 1.0 | theft: {class_weight[1]:.2f}")

    early_stop = EarlyStopping(
        monitor="val_roc_auc", patience=5, restore_best_weights=True, mode="max"
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    return model, history, X_test, y_test
