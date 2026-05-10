"""
05_app.py
Short-term Electricity Load Forecasting
Streamlit App
    Tab 1: Model Performance (test set — actual vs predicted)
    Tab 2: Forecast (recursive 24h prediction from chosen datetime)
"""

import os
import pickle
from datetime import timedelta

import holidays as hol_lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TARGET_COUNTRIES = ['HU', 'DE', 'FR', 'IT']
COUNTRY_NAMES = {
    'HU': '🇭🇺 Hungary',
    'DE': '🇩🇪 Germany',
    'FR': '🇫🇷 France',
    'IT': '🇮🇹 Italy',
}
HOL_MAP = {
    'HU': hol_lib.Hungary,
    'DE': hol_lib.Germany,
    'FR': hol_lib.France,
    'IT': hol_lib.Italy,
}

# Feature columns — must match 02_feature_engineering exactly
FEATURE_COLS = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_weekend', 'is_holiday',
    'load_lag_1h', 'load_lag_24h', 'load_lag_168h',
    'load_roll_24h_mean', 'load_roll_24h_std',
    'load_roll_168h_mean', 'load_roll_168h_std',
    'value'
]

LOOKBACK = 24    # Hours of history the LSTM expects
HORIZON  = 24    # Hours to forecast ahead
LAG_168  = 168   # History needed for lag_168h feature

PATHS = {
    'models':      'models/',
    'scalers':     'data/processed/scalers.pkl',
    'featured_df': 'data/processed/featured_df.parquet',
}

N_HOURS = 200


# ─────────────────────────────────────────────
# LOADERS (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_results(country_code: str) -> dict | None:
    """Load saved evaluation results (y_true, y_pred, MAE, RMSE) per split."""
    path = f"{PATHS['models']}{country_code}_results.npy"
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


@st.cache_data
def load_all_summaries() -> pd.DataFrame:
    """Load summary.csv with metrics for all countries."""
    path = f"{PATHS['models']}summary.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col='country')


@st.cache_data
def load_featured_df() -> pd.DataFrame:
    """Load the full feature-engineered DataFrame for forecast input."""
    return pd.read_parquet(PATHS['featured_df'])


@st.cache_resource
def load_model(country_code: str):
    """Load trained LSTM model for a given country."""
    path = f"{PATHS['models']}{country_code}_lstm.keras"
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_scalers() -> dict:
    """Load per-country MinMaxScalers."""
    with open(PATHS['scalers'], 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def denormalize(y_scaled: np.ndarray, scaler) -> np.ndarray:
    """Inverse transform normalized values back to MW."""
    n_features = scaler.scale_.shape[0]
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, -1] = y_scaled   # 'value' is the last column
    return scaler.inverse_transform(dummy)[:, -1]


def compute_time_features(dt: pd.Timestamp) -> dict:
    """Compute all time-based and cyclic features for a given timestamp."""
    return {
        'hour':       dt.hour,
        'dayofweek':  dt.dayofweek,
        'month':      dt.month,
        'quarter':    (dt.month - 1) // 3 + 1,
        'dayofyear':  dt.dayofyear,
        'is_weekend': int(dt.dayofweek >= 5),
        'hour_sin':   np.sin(2 * np.pi * dt.hour      / 24),
        'hour_cos':   np.cos(2 * np.pi * dt.hour      / 24),
        'dow_sin':    np.sin(2 * np.pi * dt.dayofweek / 7),
        'dow_cos':    np.cos(2 * np.pi * dt.dayofweek / 7),
        'month_sin':  np.sin(2 * np.pi * dt.month     / 12),
        'month_cos':  np.cos(2 * np.pi * dt.month     / 12),
    }


# ─────────────────────────────────────────────
# FORECAST LOGIC
# ─────────────────────────────────────────────
def run_forecast(df_country: pd.DataFrame,
                 chosen_dt: pd.Timestamp,
                 model,
                 scaler,
                 country_code: str,
                 horizon: int = HORIZON) -> tuple[list, list]:
    """
    Direct 24h forecast starting from chosen_dt.
    Takes a single window ending at chosen_dt and predicts all 24h at once.
    """
    # Need 24 rows of history up to AND INCLUDING chosen_dt
    history = df_country[df_country.index <= chosen_dt].tail(LOOKBACK).copy()

    if len(history) < LOOKBACK:
        raise ValueError(
            f"Not enough history before {chosen_dt}. "
            f"Need at least {LOOKBACK} hours, got {len(history)}."
        )

    # Initial X_window ends at chosen_dt
    X_input = history[FEATURE_COLS].values.reshape(1, LOOKBACK, len(FEATURE_COLS))

    # 1. Predict all 24 steps at once
    y_pred_scaled = model.predict(X_input, verbose=0) # shape (1, 24)
    y_pred_scaled = np.clip(y_pred_scaled, 0.0, 1.0).flatten()

    # 2. Denormalize all to MW
    y_pred_mw = denormalize(y_pred_scaled, scaler)

    # 3. Generate timestamps
    timestamps = [chosen_dt + timedelta(hours=step + 1) for step in range(horizon)]

    return timestamps, y_pred_mw.tolist()



# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_performance(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     country_code: str,
                     n_hours: int) -> plt.Figure:
    """Tab 1: Prediction vs. Actual on test set."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true[:n_hours, 0], label='Actual',    linewidth=2, alpha=0.9)
    ax.plot(y_pred[:n_hours, 0], label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_title(
        f'{COUNTRY_NAMES[country_code]}: Prediction vs. Actual '
        f'(first {n_hours} hours of test set)', fontsize=13
    )
    ax.set_xlabel('Hour')
    ax.set_ylabel('Load (MW)')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_forecast(timestamps: list,
                  predictions: list,
                  country_code: str,
                  chosen_dt: pd.Timestamp,
                  actual_timestamps=None,
                  actual_mw=None) -> plt.Figure:
    """Tab 2: 24h recursive forecast + actual (if available)."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(timestamps, predictions,
            label='Forecast', linewidth=2, color='darkorange',
            marker='o', markersize=3)

    if actual_timestamps is not None and actual_mw is not None:
        ax.plot(actual_timestamps, actual_mw,
                label='Actual', linewidth=2, color='steelblue', alpha=0.85)

    ax.axvline(x=chosen_dt, color='gray', linestyle='--',
               alpha=0.7, label='Forecast start')

    ax.set_title(
        f'{COUNTRY_NAMES[country_code]}: 24h Forecast from '
        f'{chosen_dt.strftime("%Y-%m-%d %H:%M")} UTC', fontsize=13
    )
    ax.set_xlabel('Datetime (UTC)')
    ax.set_ylabel('Load (MW)')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title='Electricity Load Forecasting',
        page_icon='⚡',
        layout='wide'
    )

    st.title('⚡ Short-term Electricity Load Forecasting')
    st.caption('LSTM-based hourly electricity consumption forecasting — HU, DE, FR, IT')
    st.divider()

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.header('Settings')
        country = st.selectbox(
            label='Select Country',
            options=TARGET_COUNTRIES,
            format_func=lambda x: COUNTRY_NAMES[x]
        )
        st.divider()
        st.caption('Model: LSTM(64) → Dense(32) → Dense(1)')
        st.caption('Loss: MAE | Optimizer: Adam')
        st.caption('Lookback: 24h | Forecast horizon: 24h')
        st.caption('Data available: 2019-01-08 → 2025-09-30')

    # ── Tabs ─────────────────────────────────
    tab1, tab2 = st.tabs(['📊 Model Performance', '🔮 Forecast'])

    # ════════════════════════════════════════
    # TAB 1: MODEL PERFORMANCE
    # ════════════════════════════════════════
    with tab1:
        results = load_results(country)

        if results is None:
            st.error(
                f"No results found for {COUNTRY_NAMES[country]}. "
                f"Run `03_model.py` first."
            )
            st.stop()

        st.subheader(f'Model Performance — {COUNTRY_NAMES[country]}')

        col1, col2, col3, col4 = st.columns(4)
        test_mae  = results['test']['mae']
        test_rmse = results['test']['rmse']
        train_mae = results['train']['mae']
        val_mae   = results['val']['mae']

        col1.metric('Test MAE',  f"{test_mae:.1f} MW")
        col2.metric('Test RMSE', f"{test_rmse:.1f} MW")
        col3.metric('Val MAE',   f"{val_mae:.1f} MW",
                    delta=f"{val_mae - test_mae:.1f} MW vs test",
                    delta_color='inverse')
        col4.metric('Train MAE', f"{train_mae:.1f} MW",
                    delta=f"{train_mae - test_mae:.1f} MW vs test",
                    delta_color='inverse')

        st.divider()

        st.subheader('Prediction vs. Actual (Test Set)')
        n_hours = st.slider('Hours to display', 24, 500, N_HOURS, 24)
        fig = plot_performance(
            results['test']['y_true_mw'],
            results['test']['y_pred_mw'],
            country, n_hours
        )
        st.pyplot(fig)
        st.caption('Blue = actual consumption, dashed = model forecast.')

        st.divider()

        st.subheader('All Countries — Summary')
        df_summary = load_all_summaries()

        if df_summary.empty:
            st.info('summary.csv not found. Run `03_model.py` for all countries first.')
        else:
            def highlight_selected(row):
                if row.name == country:
                    return ['background-color: #31333F; font-weight: bold'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_summary.style.apply(highlight_selected, axis=1).format({
                    'test_mae':      '{:.1f} MW',
                    'test_rmse':     '{:.1f} MW',
                    'best_val_loss': '{:.4f}',
                }),
                use_container_width=True
            )

    # ════════════════════════════════════════
    # TAB 2: FORECAST
    # ════════════════════════════════════════
    with tab2:
        st.subheader(f'24h Forecast — {COUNTRY_NAMES[country]}')
        st.caption(
            'Select a start datetime. The model will forecast the next 24 hours. '
            'Where actual data exists, it will be shown for comparison. '
            'Data available up to **2025-09-30**.'
        )

        col_date, col_hour = st.columns([3, 1])

        with col_date:
            chosen_date = st.date_input(
                label='Start date',
                value=pd.Timestamp('2025-09-01').date(),
                min_value=pd.Timestamp('2019-07-10').date(),   # 168h after data start
                max_value=pd.Timestamp('2025-09-29').date(),
            )

        with col_hour:
            chosen_hour = st.selectbox(
                label='Start hour (UTC)',
                options=list(range(24)),
                index=8,
                format_func=lambda x: f"{x:02d}:00"
            )

        chosen_dt = pd.Timestamp(
            year=chosen_date.year, month=chosen_date.month, day=chosen_date.day,
            hour=chosen_hour, tz='UTC'
        )

        st.info(f"Forecasting 24 hours from: **{chosen_dt.strftime('%Y-%m-%d %H:%M')} UTC**")

        if st.button('🔮 Run Forecast', type='primary'):

            model   = load_model(country)
            scalers = load_scalers()
            df_full = load_featured_df()

            if model is None:
                st.error(f"Model not found for {COUNTRY_NAMES[country]}. Run `03_model.py` first.")
                st.stop()

            if country not in scalers:
                st.error(f"Scaler not found for {COUNTRY_NAMES[country]}.")
                st.stop()

            df_country = df_full[df_full['countrycode'] == country].copy()

            with st.spinner('Running forecast...'):
                try:
                    timestamps, predictions = run_forecast(
                        df_country, chosen_dt,
                        model, scalers[country],
                        country_code=country
                    )
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

            # Load actual values for comparison (if available)
            actual_window = df_country[
                (df_country.index >= timestamps[0]) &
                (df_country.index <= timestamps[-1])
            ].copy()

            if len(actual_window) > 0:
                actual_mw         = denormalize(actual_window['value'].values, scalers[country])
                actual_timestamps = actual_window.index.tolist()
            else:
                actual_mw         = None
                actual_timestamps = None

            # Metrics
            col_a, col_b, col_c = st.columns(3)
            col_a.metric('Peak Forecast',    f"{max(predictions):.0f} MW")
            col_b.metric('Min Forecast',     f"{min(predictions):.0f} MW")
            col_c.metric('Average Forecast', f"{np.mean(predictions):.0f} MW")

            # Forecast MAE vs actual
            if actual_mw is not None:
                n = min(len(actual_mw), len(predictions))
                forecast_mae = mean_absolute_error(actual_mw[:n], predictions[:n])
                st.success(f"📊 Forecast MAE vs actual: **{forecast_mae:.1f} MW**")
            else:
                st.info("ℹ️ No actual data available for this period.")

            st.divider()

            fig = plot_forecast(
                timestamps, predictions, country, chosen_dt,
                actual_timestamps=actual_timestamps,
                actual_mw=actual_mw
            )
            st.pyplot(fig)
            st.caption(
                'Recursive 24-step forecast. Each predicted value feeds into the next step. '
                'Uncertainty increases with forecast horizon.'
            )

            st.divider()

            # Forecast table
            st.subheader('Forecast Values')
            table_data = {
                'Datetime (UTC)':      [t.strftime('%Y-%m-%d %H:%M') for t in timestamps],
                'Predicted Load (MW)': [f"{p:.0f}" for p in predictions],
            }
            if actual_mw is not None:
                n = min(len(actual_mw), len(predictions))
                padded_actual = [f"{v:.0f}" for v in actual_mw[:n]]
                padded_actual += ['—'] * (len(predictions) - n)
                table_data['Actual Load (MW)'] = padded_actual

            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True
            )


if __name__ == '__main__':
    main()
