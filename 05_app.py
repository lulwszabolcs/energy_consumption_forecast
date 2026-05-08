"""
05_app.py
Short-term Electricity Load Forecasting
Streamlit App — Tab 1: Model Performance
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
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

PATHS = {
    'models':  'models/',
    'scalers': 'data/processed/scalers.pkl',
}

N_HOURS = 200   # Default hours to show on plot


# ─────────────────────────────────────────────
# DATA LOADING (cached for performance)
# ─────────────────────────────────────────────
@st.cache_data
def load_results(country_code: str) -> dict | None:
    """Load saved evaluation results for a given country."""
    results_path = f"{PATHS['models']}{country_code}_results.npy"
    if not os.path.exists(results_path):
        return None
    return np.load(results_path, allow_pickle=True).item()


@st.cache_data
def load_all_summaries() -> pd.DataFrame:
    """Load summary.csv with MAE/RMSE per country."""
    summary_path = f"{PATHS['models']}summary.csv"
    if not os.path.exists(summary_path):
        return pd.DataFrame()
    return pd.read_csv(summary_path, index_col='country')


# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
def plot_prediction_vs_actual(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               country_code: str,
                               n_hours: int = N_HOURS) -> plt.Figure:
    """Generate Prediction vs. Actual matplotlib figure."""
    y_true = y_true[:n_hours]
    y_pred = y_pred[:n_hours]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label='Actual',    linewidth=2, alpha=0.9)
    ax.plot(y_pred, label='Predicted', linewidth=2, alpha=0.8, linestyle='--')

    ax.set_title(
        f'{COUNTRY_NAMES[country_code]}: Prediction vs. Actual '
        f'(first {n_hours} hours of test set)',
        fontsize=13
    )
    ax.set_xlabel('Hour')
    ax.set_ylabel('Load (MW)')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
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

        n_hours = st.slider(
            label='Hours to display',
            min_value=24,
            max_value=500,
            value=N_HOURS,
            step=24
        )

        st.divider()
        st.caption('Model: LSTM(64) → Dense(32) → Dense(1)')
        st.caption('Loss: MAE | Optimizer: Adam')
        st.caption('Lookback: 24h | Forecast: 1h')

    # ── Load results ─────────────────────────
    results = load_results(country)

    if results is None:
        st.error(
            f"No results found for {COUNTRY_NAMES[country]}. "
            f"Run `03_model.py` first to train the model."
        )
        st.stop()

    # ── Metrics row ──────────────────────────
    st.subheader(f'Model Performance — {COUNTRY_NAMES[country]}')

    col1, col2, col3, col4 = st.columns(4)

    test_mae  = results['test']['mae']
    test_rmse = results['test']['rmse']
    train_mae = results['train']['mae']
    val_mae   = results['val']['mae']

    col1.metric(label='Test MAE',  value=f"{test_mae:.1f} MW")
    col2.metric(label='Test RMSE', value=f"{test_rmse:.1f} MW")
    col3.metric(
        label='Val MAE',
        value=f"{val_mae:.1f} MW",
        delta=f"{val_mae - test_mae:.1f} MW vs test",
        delta_color='inverse'
    )
    col4.metric(
        label='Train MAE',
        value=f"{train_mae:.1f} MW",
        delta=f"{train_mae - test_mae:.1f} MW vs test",
        delta_color='inverse'
    )

    st.divider()

    # ── Prediction vs Actual plot ─────────────
    st.subheader('Prediction vs. Actual (Test Set)')

    y_true = results['test']['y_true_mw']
    y_pred = results['test']['y_pred_mw']

    fig = plot_prediction_vs_actual(y_true, y_pred, country, n_hours)
    st.pyplot(fig)

    st.caption(
        f'Showing first {n_hours} hours of the test set. '
        f'Blue = actual consumption, dashed = model forecast.'
    )

    st.divider()

    # ── All countries summary table ───────────
    st.subheader('All Countries — Summary')

    df_summary = load_all_summaries()

    if df_summary.empty:
        st.info('summary.csv not found. Run `03_model.py` for all countries first.')
    else:
        # Highlight selected country
        def highlight_selected(row):
            return ['background-color: #e8f4fd' if row.name == country else '' for _ in row]

        st.dataframe(
            df_summary.style.apply(highlight_selected, axis=1).format({
                'test_mae':      '{:.1f} MW',
                'test_rmse':     '{:.1f} MW',
                'best_val_loss': '{:.4f}',
            }),
            use_container_width=True
        )


if __name__ == '__main__':
    main()
