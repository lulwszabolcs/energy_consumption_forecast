import pandas as pd
import numpy as np
import holidays

LAG_HOURS        = [1, 24, 168]
ROLLING_WINDOWS  = [24, 168]
LOOKBACK         = 24    # Hours to look back
FORECAST_HORIZON = 1     # Hours to forecast

HOLIDAY_MAP = {
    'HU': holidays.Hungary,
    'DE': holidays.Germany,
    'FR': holidays.France,
    'IT': holidays.Italy,
}

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    years = range(df.index.year.min(), df.index.year.max() + 1)

    # Pre-compute holiday sets per country
    holiday_sets = {
        cc: set(hol_cls(years=list(years)).keys())
        for cc, hol_cls in HOLIDAY_MAP.items()
    }

    date_col = pd.to_datetime(df['dateshort']).dt.date

    df['is_holiday'] = [
        int(date_col.iloc[i] in holiday_sets.get(df['countrycode'].iloc[i], set()))
        for i in range(len(df))
    ]
    return df

def add_lag_features(df: pd.DataFrame, lags: list = LAG_HOURS) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f'load_lag_{lag}h'] = df.groupby('countrycode')['value'].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby('countrycode')['value']
    for w in windows:
        df[f'load_roll_{w}h_mean'] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f'load_roll_{w}h_std']  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
    return df