# 01 — Exploratory Data Analysis (EDA)

## Overview
This notebook loads the raw electricity consumption dataset, performs cleaning, and explores the data through descriptive statistics and visualizations. The output is a cleaned parquet file used in the next step.

---

## Input
- `data/raw/electricity_raw.csv` — Original Kaggle dataset (~2M rows, 2019–2025)

## Output
- `data/processed/clean.parquet` — Cleaned, indexed DataFrame ready for feature engineering

---

## Steps

### 1. Load Raw Data
Loads the raw CSV into a Pandas DataFrame. Sets `dateutc` as the datetime index (timezone-aware). Converts column names to lowercase.

### 2. Data Cleaning
- Removes redundant columns: `MeasureItem`, `Cov_ratio`
- Converts `dateutc` to datetime index
- Reduces `timefrom` / `timeto` to hour-only format (`HH:MM:SS`)
- Adds `year` column extracted from the index

### 3. Country Filtering
Filters the dataset to the four target countries:
- 🇭🇺 Hungary (HU)
- 🇩🇪 Germany (DE)
- 🇫🇷 France (FR)
- 🇮🇹 Italy (IT)

### 4. Descriptive Statistics
- Row counts and date ranges per country
- Missing value analysis (`timefrom`, `timeto` null counts)
- Basic statistics: mean, min, max, std of `value` (MW) per country

### 5. Save Cleaned Data
Saves the cleaned DataFrame to `data/processed/clean.parquet` for use in `02_feature_engineering.ipynb`.

---

## Key Columns After Cleaning

| Column | Type | Description |
|---|---|---|
| `dateshort` | str | Date only (YYYY-MM-DD) |
| `timefrom` | str | Hour start (HH:MM:SS) |
| `timeto` | str | Hour end (HH:MM:SS) |
| `countrycode` | str | Country identifier (HU, DE, FR, IT) |
| `value` | float64 | Electricity load (MW) |
| `value_scaleto100` | float64 | Scaled load value |
| `year` | int64 | Year extracted from index |

---

## Notes
- `timefrom` / `timeto` have ~35,000 missing values — these are handled in feature engineering
- `value_scaleto100` is kept for reference but not used in modeling
- Index is timezone-aware UTC (`+00:00`)
