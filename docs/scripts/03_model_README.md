# 03 — Model Training (`03_model.py`)

## Summary
This script handles the full LSTM training pipeline for electricity load forecasting. It loads the preprocessed datasets from the feature engineering step, builds a separate LSTM model for each country, trains and evaluates each model, and saves the results. Visualization is handled separately in `04_visualize.py`. Run this script before any visualization or app usage.

```bash
python 03_model.py
```

---

## Input
- `data/processed/datasets.npy` — X/y sequences split into train/val/test per country
- `data/processed/scalers.pkl` — Per-country MinMaxScalers for denormalization

## Output
- `models/{CC}_lstm.keras` — Final trained model per country
- `models/{CC}_best.keras` — Best checkpoint saved during training
- `models/{CC}_history.npy` — Training loss history per epoch
- `models/{CC}_results.npy` — Evaluation results (y_true, y_pred, MAE, RMSE) per split
- `models/summary.csv` — Final metrics table across all countries
- `models/training.log` — Full training log with timestamps

---

## Model Architecture

```
Input: (24, 15)        ← 24-hour lookback, 15 features
    ↓
LSTM(64)               ← learns temporal patterns
Dropout(0.2)           ← prevents overfitting
    ↓
Dense(32, ReLU)        ← processes LSTM output
Dropout(0.2)
    ↓
Dense(1)               ← single output, no activation (free regression)
    ↓
Output: predicted load (normalized 0–1)
```

**Loss:** MAE | **Optimizer:** Adam (default lr) | **Epochs:** 100 | **Batch size:** 32 | **Early stopping patience:** 10

---

## Methods

### `load_datasets(path: str) -> dict`
Loads the preprocessed X/y dataset dictionary from `datasets.npy`.

- **Input:** Path to `datasets.npy`
- **Output:** Dictionary with structure `{'HU': {'X_train': ..., 'y_train': ..., 'X_val': ..., 'y_val': ..., 'X_test': ..., 'y_test': ...}, 'DE': {...}, ...}`
- **Note:** Uses `allow_pickle=True` and `.item()` to unwrap the numpy array into a plain Python dictionary

---

### `load_scalers(path: str) -> dict`
Loads the per-country MinMaxScaler objects from `scalers.pkl`.

- **Input:** Path to `scalers.pkl`
- **Output:** Dictionary `{'HU': MinMaxScaler, 'DE': MinMaxScaler, ...}`
- **Note:** Scalers are needed in `evaluate_model()` to denormalize predictions back to MW

---

### `build_lstm(input_shape: tuple, config: dict) -> tf.keras.Model`
Builds and compiles the LSTM model architecture.

- **Input:**
  - `input_shape` — Tuple `(lookback, n_features)`, e.g. `(24, 15)`
  - `config` — Dictionary with model hyperparameters (`lstm_units`, `dense_units`, `dropout_rate`, `loss`, `optimizer`)
- **Output:** Compiled but untrained `tf.keras.Model`
- **Note:** The final `Dense(1)` layer has no activation function — the output is a free regression value, not bounded to [0, 1]

---

### `train_model(model, splits, config, country_code) -> History`
Trains the LSTM model on the training split, monitoring validation loss throughout.

- **Input:**
  - `model` — Compiled Keras model from `build_lstm()`
  - `splits` — Country-specific dictionary with `X_train`, `y_train`, `X_val`, `y_val`
  - `config` — Training hyperparameters (`epochs`, `batch_size`, `patience`)
  - `country_code` — Used for checkpoint filename and logging
- **Output:** Keras `History` object containing per-epoch loss values
- **Callbacks:**
  - `EarlyStopping(patience=10)` — Stops training if validation loss does not improve for 10 consecutive epochs. Restores the best weights automatically.
  - `ModelCheckpoint` — Saves the best model to `models/{CC}_best.keras` during training
- **Note:** Training may stop before 100 epochs if Early Stopping triggers

---

### `denormalize(y_scaled: np.ndarray, scaler) -> np.ndarray`
Converts normalized model predictions (0–1) back to real MW values.

- **Input:**
  - `y_scaled` — 1D numpy array of normalized values
  - `scaler` — Country-specific MinMaxScaler
- **Output:** 1D numpy array of denormalized values in MW
- **Note:** The scaler was fit on all 15 feature columns with `value` as the last column. A dummy array of zeros is constructed and the normalized values are placed in the last column before calling `inverse_transform()`.

---

### `evaluate_model(model, splits, scaler, country_code) -> dict`
Evaluates the trained model on train, validation, and test splits. Denormalizes predictions to MW before computing metrics.

- **Input:**
  - `model` — Trained Keras model
  - `splits` — Full country splits dictionary including test data
  - `scaler` — Country-specific MinMaxScaler for denormalization
  - `country_code` — Used for logging
- **Output:** Nested dictionary with structure:
  ```python
  {
    'train': {'mae': float, 'rmse': float, 'y_true_mw': array, 'y_pred_mw': array},
    'val':   {'mae': float, 'rmse': float, 'y_true_mw': array, 'y_pred_mw': array},
    'test':  {'mae': float, 'rmse': float, 'y_true_mw': array, 'y_pred_mw': array},
  }
  ```
- **Metrics:**
  - **MAE** — Mean Absolute Error in MW (average prediction error)
  - **RMSE** — Root Mean Squared Error in MW (penalizes large errors more)

---

### `save_outputs(model, history, results, country_code) -> None`
Saves all training outputs for a given country to the `models/` directory.

- **Input:**
  - `model` — Trained Keras model
  - `history` — Training history from `train_model()`
  - `results` — Evaluation results from `evaluate_model()`
  - `country_code` — Used to name output files
- **Output files:**
  - `models/{CC}_lstm.keras` — Full trained model (architecture + weights)
  - `models/{CC}_history.npy` — Per-epoch train/val loss values
  - `models/{CC}_results.npy` — y_true, y_pred, MAE, RMSE per split
- **Note:** Results are saved separately so `04_visualize.py` and `05_app.py` can load them without rerunning training

## Notes
- Each country gets its own independently trained LSTM model
- Training logs are written to both console and `models/training.log`
- If a country is missing from `datasets.npy` it is skipped with a warning
- Visualization is intentionally separated into `04_visualize.py` to keep this script focused on training
