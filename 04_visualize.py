"""
04_visualize.py
Short-term Electricity Load Forecasting
Visualization: Prediction vs. Actual
Loads saved results from 03_model.py and generates timestamped PNG plots.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TARGET_COUNTRIES = ['HU', 'DE', 'FR', 'IT']

PATHS = {
    'models': 'models/',
    'plots':  'plots/',
}

N_HOURS = 200   # Number of hours to show on the plot


# ─────────────────────────────────────────────
# 1. LOAD RESULTS
# ─────────────────────────────────────────────
def load_results(country_code: str) -> dict:
    """Load saved evaluation results for a given country."""
    results_path = f"{PATHS['models']}{country_code}_results.npy"

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Results not found for {country_code} at {results_path}. "
            f"Run 03_model.py first."
        )

    results = np.load(results_path, allow_pickle=True).item()
    print(f"{country_code} | Results loaded from {results_path}")
    return results


# ─────────────────────────────────────────────
# 2. PLOT PREDICTION VS ACTUAL
# ─────────────────────────────────────────────
def plot_prediction_vs_actual(results: dict,
                               country_code: str,
                               run_dir: str,
                               n_hours: int = N_HOURS) -> None:
    """
    Plot actual vs predicted load (MW) on test set.
    Uses the first step of each 24h forecast window for comparison.
    Saves PNG to timestamped run directory.
    """
    # y_true/y_pred now have shape (N, 24)
    # We take the first predicted hour of each window for a continuous 1-step plot
    y_true = results['test']['y_true_mw'][:n_hours, 0]
    y_pred = results['test']['y_pred_mw'][:n_hours, 0]

    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label='Actual',    linewidth=2, alpha=0.9)
    plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.8, linestyle='--')

    plt.title(
        f'{country_code}: Prediction vs. Actual (first {n_hours} hours of test set)',
        fontsize=14
    )
    plt.xlabel('Hour')
    plt.ylabel('Load (MW)')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save to timestamped run directory
    plot_path = f"{run_dir}{country_code}_prediction.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()   # Close figure to free memory (no plt.show() in script)

    print(f"{country_code} | Plot saved → {plot_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{PATHS['plots']}{timestamp}/"
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 50)
    print(f"Visualization run: {timestamp}")
    print(f"Output directory:  {run_dir}")
    print("=" * 50)

    success = []
    failed  = []

    for cc in TARGET_COUNTRIES:
        try:
            results = load_results(cc)
            plot_prediction_vs_actual(results, cc, run_dir)
            success.append(cc)
        except FileNotFoundError as e:
            print(f"SKIPPED {cc}: {e}")
            failed.append(cc)

    # Summary
    print("\n" + "=" * 50)
    print(f"Done! Plots saved to: {run_dir}")
    print(f"  Success: {success}")
    if failed:
        print(f"  Skipped: {failed} (run 03_model.py first)")
    print("=" * 50)


if __name__ == "__main__":
    main()
