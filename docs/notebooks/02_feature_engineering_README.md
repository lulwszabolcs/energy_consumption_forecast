# 02 — Feature Engineering

## Overview
This notebook transforms the cleaned electricity dataset into model-ready input for the LSTM. It enriches the data with time-based features, holiday flags, lag values, and rolling aggregates. The output is a normalized dataset split into train/val/test sequences per country.

---

## Input
- `data/processed/clean.parquet` — Cleaned data from `01_eda.ipynb`
- `utils.py` — Shared feature engineering functions (`add_holiday_features`, `add_lag_features`, `add_rolling_features`)

## Output
- `data/processed/featured_df.parquet` — Full feature-enriched, normalized DataFrame
- `data/processed/datasets.npy` — X/y sequences split into train/val/test per country
[- `data/processed/scalers.pkl` — Per-country MinMaxScalers (needed for denormalization in later steps)
]()
---

## Steps

### 1. Load Cleaned Data
Loads `clean.parquet`, filters to target countries (HU, DE, FR, IT), and sorts by datetime index.

### 2. Time-based Features
Extracts temporal information from the datetime index:

| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23) |
| `dayofweek` | Day of week (0=Monday, 6=Sunday) |
| `month` | Month (1–12) |
| `quarter` | Quarter (1–4) |
| `dayofyear` | Day of year (1–365) |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |
| `hour_sin`, `hour_cos` | Cyclic encoding of hour |
| `dow_sin`, `dow_cos` | Cyclic encoding of day of week |
| `month_sin`, `month_cos` | Cyclic encoding of month |

**Why cyclic encoding?** Raw hour values (0–23) imply that hour 23 and hour 0 are far apart. Sin/cos encoding places them on a circle so the model correctly treats them as adjacent.

### 3. Holiday Features
Adds an `is_holiday` binary flag (1 = public holiday, 0 = normal day) using country-specific holiday calendars via the `holidays` library. Handled by `utils.add_holiday_features()`.

### 4. Lag Features
Adds past load values as predictors. Handled by `utils.add_lag_features()`.

| Feature | Description |
|---|---|
| `load_lag_1h` | Load 1 hour ago |
| `load_lag_24h` | Load 24 hours ago (same hour yesterday) |
| `load_lag_168h` | Load 168 hours ago (same hour last week) |

### 5. Rolling Aggregate Features
Adds smoothed load statistics over recent windows. Handled by `utils.add_rolling_features()`. A `shift(1)` is applied to prevent data leakage.

| Feature | Description |
|---|---|
| `load_roll_24h_mean` | 24h rolling mean |
| `load_roll_24h_std` | 24h rolling standard deviation |
| `load_roll_168h_mean` | 168h rolling mean |
| `load_roll_168h_std` | 168h rolling standard deviation |

### 6. Drop NaNs
Lag features introduce NaN values at the start of each country's time series (first 168 rows). These are dropped before normalization.

### 7. Normalization
Applies `MinMaxScaler` per country to scale all feature values to the range [0, 1]. Per-country scaling is necessary because load magnitudes differ significantly (HU ~4,000 MW vs. DE ~50,000 MW). Scalers are saved to `scalers.pkl` for use during inference.

### 8. Sequence Building
Converts the DataFrame into 3D tensors for LSTM input:
- **X shape:** `(samples, 24, 15)` — 24-hour lookback window, 15 features
- **y shape:** `(samples,)` — target load value at hour t+1

### 9. Train / Val / Test Split
Time-ordered split (no shuffling) per country:
- **Train:** 80%
- **Validation:** 10%
- **Test:** 10%

---

## Feature Summary

| Category | Features | Count |
|---|---|---|
| Cyclic time | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` | 6 |
| Binary flags | `is_weekend`, `is_holiday` | 2 |
| Lag values | `load_lag_1h`, `load_lag_24h`, `load_lag_168h` | 3 |
| Rolling stats | `load_roll_24h/168h_mean/std` | 4 |
| **Total** | | **15** |

---

## Notes
- Sequences are built per country separately to avoid mixing countries in the same window
- The `value` column (target) is included in the scaler fit as the last column — this is required for correct denormalization in `03_model.py`
- Saved `datasets.npy` is a Python dictionary: `{'HU': {'X_train': ..., 'y_train': ..., ...}, ...}`
