"""
03_model.py
Short-term Electricity Load Forecasting
LSTM Model: Build, Train, Evaluate, Save
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TARGET_COUNTRIES = ['HU', 'DE', 'FR', 'IT']

MODEL_CONFIG = {
    'lstm_units':   64,
    'dense_units':  32,
    'dropout_rate': 0.2,
    'epochs':       100,
    'batch_size':   32,
    'patience':     10,
    'loss':         'mae',
    'optimizer':    'adam',
}

PATHS = {
    'datasets': 'data/processed/datasets.npy',
    'scalers':  'data/processed/scalers.pkl',
    'models':   'models/',
}

os.makedirs(PATHS['models'], exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('models/training.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────
def load_datasets(path: str) -> dict:
    """Load preprocessed X/y splits per country from feature engineering step."""
    datasets = np.load(path, allow_pickle=True).item()
    log.info(f"Datasets loaded from {path}")
    log.info(f"Available countries: {list(datasets.keys())}")
    return datasets


def load_scalers(path: str) -> dict:
    """Load per-country MinMaxScalers for denormalization."""
    with open(path, 'rb') as f:
        scalers = pickle.load(f)
    log.info(f"Scalers loaded from {path}")
    return scalers


# ─────────────────────────────────────────────
# 2. BUILD MODEL
# ─────────────────────────────────────────────
def build_lstm(input_shape: tuple, config: dict) -> tf.keras.Model:
    """
    Build LSTM model.
    Architecture:
        Input(lookback, n_features)
        → LSTM(64) → Dropout(0.2)
        → Dense(32) → Dropout(0.2)
        → Dense(1)  [no activation — free regression output]
    """

    "Group of layers into a model using Sequential"
    model = Sequential([
        LSTM(
            config['lstm_units'],
            input_shape=input_shape
        ),
        Dropout(config['dropout_rate']),

        Dense(config['dense_units']),
        Dropout(config['dropout_rate']),

        Dense(1)   # No activation — free regression output
    ])

    model.compile(
        optimizer=config['optimizer'],
        loss=config['loss'],
        metrics=['mae']
    )

    return model


# ─────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(model: tf.keras.Model,
                splits: dict,
                config: dict,
                country_code: str) -> tf.keras.callbacks.History:
    """
    Train LSTM with EarlyStopping and ModelCheckpoint.
    Validation data is monitored throughout training.
    """
    checkpoint_path = f"{PATHS['models']}{country_code}_best.keras"

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]

    log.info(f"{country_code} | Training started | epochs={config['epochs']}, batch_size={config['batch_size']}")

    history = model.fit(
        splits['X_train'], splits['y_train'],
        validation_data=(splits['X_val'], splits['y_val']),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    log.info(f"{country_code} | Training finished | best val_loss={min(history.history['val_loss']):.4f}")

    return history


# ─────────────────────────────────────────────
# 4. EVALUATE MODEL
# ─────────────────────────────────────────────
def denormalize(y_scaled: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse transform normalized values back to MW.
    Scaler was fit on all feature columns — 'value' was the last column.
    We build a dummy array to use inverse_transform correctly.
    """
    n_features = scaler.scale_.shape[0]
    dummy = np.zeros((len(y_scaled), n_features)) # dummy arrays with zeros to use inverse_transform correctly
    dummy[:, -1] = y_scaled   # 'value' is the last column
    return scaler.inverse_transform(dummy)[:, -1]


def evaluate_model(model: tf.keras.Model,
                   splits: dict,
                   scaler,
                   country_code: str) -> dict:
    """
    Evaluate on train / val / test splits.
    Denormalizes predictions to MW for interpretable MAE and RMSE.
    Returns results dict with metrics and raw predictions per split.
    """
    results = {}

    for split_name in ['train', 'val', 'test']:
        X             = splits[f'X_{split_name}']
        y_true_scaled = splits[f'y_{split_name}']

        # Predict (normalized output)
        y_pred_scaled = model.predict(X, verbose=0).flatten()

        # Denormalize both to MW
        y_true_mw = denormalize(y_true_scaled, scaler)
        y_pred_mw = denormalize(y_pred_scaled, scaler)

        mae  = mean_absolute_error(y_true_mw, y_pred_mw)
        rmse = np.sqrt(mean_squared_error(y_true_mw, y_pred_mw))

        results[split_name] = {
            'mae':       mae,
            'rmse':      rmse,
            'y_true_mw': y_true_mw,
            'y_pred_mw': y_pred_mw,
        }

        log.info(f"{country_code} | {split_name:5s} | MAE={mae:.1f} MW | RMSE={rmse:.1f} MW")

    return results


# ─────────────────────────────────────────────
# 5. SAVE MODEL & HISTORY
# ─────────────────────────────────────────────
def save_outputs(model: tf.keras.Model,
                 history: tf.keras.callbacks.History,
                 results: dict,
                 country_code: str) -> None:
    """
    Save trained model, training history, and evaluation results.
    Results are saved so 04_visualize.py can load them without re-running training.
    """
    model_path   = f"{PATHS['models']}{country_code}_lstm.keras"
    history_path = f"{PATHS['models']}{country_code}_history.npy"
    results_path = f"{PATHS['models']}{country_code}_results.npy"

    model.save(model_path)
    np.save(history_path, history.history)
    np.save(results_path, results)

    log.info(f"{country_code} | Model saved   → {model_path}")
    log.info(f"{country_code} | History saved → {history_path}")
    log.info(f"{country_code} | Results saved → {results_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=" * 50)
    log.info("Electricity Load Forecasting — Model Training")
    log.info("=" * 50)

    # Load data
    datasets = load_datasets(PATHS['datasets'])
    scalers  = load_scalers(PATHS['scalers'])

    summary = []

    for cc in TARGET_COUNTRIES:
        if cc not in datasets:
            log.warning(f"{cc} | Not found in datasets, skipping")
            continue

        log.info(f"\n{'='*40}")
        log.info(f"Country: {cc}")
        log.info(f"{'='*40}")

        splits = datasets[cc]

        # Input shape: (lookback=24, n_features)
        input_shape = (splits['X_train'].shape[1], splits['X_train'].shape[2])
        log.info(f"{cc} | Input shape: {input_shape}")
        log.info(f"{cc} | Train: {len(splits['X_train']):,} | Val: {len(splits['X_val']):,} | Test: {len(splits['X_test']):,}")

        # 1. Build
        model = build_lstm(input_shape, MODEL_CONFIG)
        model.summary(print_fn=lambda x: log.info(x))

        # 2. Train
        history = train_model(model, splits, MODEL_CONFIG, cc)

        # 3. Evaluate
        results = evaluate_model(model, splits, scalers[cc], cc)

        # 4. Save (no visualization here — see 04_visualize.py)
        save_outputs(model, history, results, cc)

        # 5. Summary row
        summary.append({
            'country':       cc,
            'test_mae':      results['test']['mae'],
            'test_rmse':     results['test']['rmse'],
            'epochs_run':    len(history.history['loss']),
            'best_val_loss': min(history.history['val_loss']),
        })

    # Save final summary table
    log.info("\n" + "=" * 50)
    log.info("FINAL SUMMARY")
    log.info("=" * 50)
    df_summary = pd.DataFrame(summary).set_index('country')
    log.info(f"\n{df_summary.round(2).to_string()}")
    df_summary.to_csv(f"{PATHS['models']}summary.csv")
    log.info(f"Summary saved → {PATHS['models']}summary.csv")


if __name__ == "__main__":
    main()
