# re-run analysis with synthetic data
# Daman Dhaliwal

import pandas as pd
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import project modules
import RL
import LSTM
import autoencoder
from ivs_create import create_ivs
from spy_prices import get_spy_prices
from utils import paths


def estimate_real_parameters():
    real_prices = get_spy_prices()
    real_prices['returns'] = real_prices['spy_price'].pct_change()
    real_prices = real_prices.dropna()

    mu = real_prices['returns'].mean() * 252
    sigma = real_prices['returns'].std() * np.sqrt(252)

    # Use real IVS to get the grid shape
    real_ivs, _ = create_ivs(overwrite=False)

    # Calculate "Base Smile" (Average shape of the surface)
    avg_surface = np.mean(real_ivs, axis=0)  # (Deltas, Maturities)
    base_level = np.mean(avg_surface)
    normalized_shape = avg_surface / base_level

    return mu, sigma, normalized_shape


def generate_synthetic_data(mu, sigma_target, base_shape, n_years=10):
    n_days = n_years * 252
    dt = 1 / 252

    vols = np.zeros(n_days)
    vols[0] = sigma_target

    kappa = 2.0  # Mean reversion speed
    theta = sigma_target  # Long run vol
    xi = 0.3  # Vol of Vol
    rho = -0.7  # Leverage effect

    Z1 = np.random.normal(0, 1, n_days)
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, n_days)

    for t in range(1, n_days):
        dv = kappa * (theta - vols[t - 1]) * dt + xi * vols[t - 1] * np.sqrt(dt) * Z2[t]
        vols[t] = max(0.01, vols[t - 1] + dv)

    S = np.zeros(n_days)
    S[0] = 100.0

    for t in range(1, n_days):
        drift = mu * dt
        diffusion = vols[t - 1] * np.sqrt(dt) * Z1[t]
        S[t] = S[t - 1] * np.exp(drift - 0.5 * vols[t - 1] ** 2 * dt + diffusion)

    # Scale base surface by current volatility
    syn_ivs = vols[:, np.newaxis, np.newaxis] * base_shape[np.newaxis, :, :]

    start_date = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    syn_prices_df = pd.DataFrame({'date': dates, 'spy_price': S})
    syn_dates = dates.to_numpy()

    return syn_prices_df, syn_ivs, syn_dates


def run_synthetic_pipeline():
    mu, sigma, base_shape = estimate_real_parameters()

    syn_prices_df, syn_ivs, syn_dates = generate_synthetic_data(mu, sigma, base_shape)

    real_paths = paths()
    syn_base = os.path.join(real_paths['parent_dir'], 'Output', 'Synthetic')

    syn_paths_dict = {
        'parent_dir': real_paths['parent_dir'],
        'data_input': os.path.join(syn_base, 'Data/'),
        'data': os.path.join(syn_base, 'Data/'),
        'plots': os.path.join(syn_base, 'Plots/'),
        'tables': os.path.join(syn_base, 'Tables/'),
        'models': os.path.join(syn_base, 'Models/')
    }

    # Create synthetic directories
    for p in syn_paths_dict.values():
        os.makedirs(p, exist_ok=True)

    def mock_paths():
        print("Using Mock Paths")
        return syn_paths_dict

    def mock_create_ivs(overwrite=False):
        return syn_ivs, syn_dates

    def mock_get_spy_prices():
        return syn_prices_df

    # Inject Mock Data Functions
    RL.create_ivs = mock_create_ivs
    RL.get_spy_prices = mock_get_spy_prices
    LSTM.create_ivs = mock_create_ivs
    autoencoder.create_ivs = mock_create_ivs

    # Inject Mock Paths (This is the key fix)
    # This forces saving/loading from Output/Synthetic/Models
    RL.paths = mock_paths
    LSTM.paths = mock_paths
    autoencoder.paths = mock_paths

    ae_model, ae_scaler = autoencoder.train_autoencoder(epochs = 100)

    LSTM.train_both_predictors(n_epochs = 100)

    results_df = RL.run_experiment_grid(n_epochs = 100, network='RNNFNN')

    # Output Results
    print(results_df)

    save_path = os.path.join(syn_paths_dict['tables'], 'synthetic_pipeline_results.tex')

    latex_code = results_df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Pipeline Validation: End-to-End Performance on Synthetic Data",
        label="tab:synthetic_pipeline",
        position="H"
    )

    with open(save_path, 'w') as f:
        f.write(latex_code)

    print(f"\nSaved synthetic results to: {save_path}")


if __name__ == "__main__":
    run_synthetic_pipeline()