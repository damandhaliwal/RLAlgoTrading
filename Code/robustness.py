import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

from RL import DeepAgent, MarketEnvironment, bs_price, compute_loss
from ivs_create import create_ivs
from spy_prices import get_spy_prices
from LSTM import train_predictor
from utils import paths

def detect_regimes(prices, n_days=63):
    """
    Classifies every possible start date in history into a market regime.
    """
    n_samples = len(prices) - n_days
    regime_indices = {
        'Normal': [],
        'Bull': [],
        'Bear': [],
        'Crisis': []
    }

    returns = []
    volatilities = []

    for i in range(n_samples):
        window = prices[i: i + n_days]
        ret = (window[-1] - window[0]) / window[0]
        returns.append(ret)

        log_rets = np.log(window[1:] / window[:-1])
        vol = np.std(log_rets) * np.sqrt(252)
        volatilities.append(vol)

    vol_threshold = np.percentile(volatilities, 95)

    for i in range(n_samples):
        ret = returns[i]
        vol = volatilities[i]

        if vol > vol_threshold:
            regime_indices['Crisis'].append(i)
        elif ret > 0.05:
            regime_indices['Bull'].append(i)
        elif ret < -0.05:
            regime_indices['Bear'].append(i)
        else:
            regime_indices['Normal'].append(i)

    for k in regime_indices:
        regime_indices[k] = np.array(regime_indices[k])
        print(f"Regime '{k}': Found {len(regime_indices[k])} episodes")

    return regime_indices


def run_robustness_check(model_path, network='RNNFNN', use_predictor=False):
    print("Loading Data...")

    predictor = None
    if use_predictor:
        predictor, scaler_pred, autoencoder = train_predictor(n_epochs=0, load_autoencoder=True)
        predictor.eval()
    else:
        _, scaler_pred, autoencoder = train_predictor(n_epochs=0, load_autoencoder=True)

    autoencoder.eval()

    ivs_data, ivs_dates = create_ivs(overwrite=False)
    ivs_flat = ivs_data.reshape((ivs_data.shape[0], -1))
    ivs_scaled = scaler_pred.transform(ivs_flat)

    with torch.no_grad():
        latents_encoded = autoencoder.encoder(torch.tensor(ivs_scaled, dtype=torch.float32)).numpy()

    spy_df = get_spy_prices()
    spy_df['date'] = pd.to_datetime(spy_df['date'])

    latent_df = pd.DataFrame({'date': ivs_dates})
    latent_df['latent_idx'] = range(len(latent_df))
    merged = pd.merge(spy_df, latent_df, on='date', how='inner').sort_values('date')

    prices = merged['spy_price'].values
    latents = latents_encoded[merged['latent_idx'].values]

    nbs_point_traj = 63
    env = MarketEnvironment(prices, latents, n_days=nbs_point_traj)

    feature_dim = 24
    # Ensure input dimension matches training (5 core + 24 feature)
    model = DeepAgent(network, 5 + feature_dim, 56, 2, 0.5)

    print(f"Loading Model from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("\nClassifying Historical Market Regimes...")
    regimes = detect_regimes(prices, n_days=nbs_point_traj)

    results = {}

    for regime_name, indices in regimes.items():
        if len(indices) == 0: continue

        print(f"\n--- Testing on {regime_name} Market ({len(indices)} episodes) ---")

        # Bootstrapping logic for evaluation
        n_evals = 500
        if len(indices) < n_evals: n_evals = len(indices)

        # Sample from this specific regime
        selected_starts = np.random.choice(indices, size=n_evals, replace=True)

        batch_prices = []
        batch_latents = []
        for start in selected_starts:
            batch_prices.append(prices[start: start + nbs_point_traj])
            batch_latents.append(latents[start: start + nbs_point_traj])

        b_prices = torch.tensor(np.array(batch_prices), dtype=torch.float32).unsqueeze(-1)
        b_latents = torch.tensor(np.array(batch_latents), dtype=torch.float32)

        # --- NORMALIZATION (MUST MATCH TRAINING) ---
        initial_prices = b_prices[:, 0, 0].unsqueeze(1)
        scale_factors = 100.0 / initial_prices
        norm_prices = b_prices * scale_factors.unsqueeze(1)
        # -------------------------------------------

        strike = 100.0
        init_S_norm = norm_prices[:, 0, 0].numpy()
        init_vals = bs_price(init_S_norm, strike, nbs_point_traj / 252, 0.0, 0.2, True)

        portfolio = torch.tensor(init_vals, dtype=torch.float32)
        pos = torch.zeros(len(batch_prices))
        hidden = None

        with torch.no_grad():
            for t in range(nbs_point_traj - 1):
                p_curr = norm_prices[:, t, 0]
                p_next = norm_prices[:, t + 1, 0]
                feat = b_latents[:, t, :]

                if use_predictor:
                    feat = predictor(feat.unsqueeze(1))

                time = torch.full((len(batch_prices), 1), (nbs_point_traj - t) / nbs_point_traj)

                state = torch.cat([feat, p_curr.unsqueeze(1), time, portfolio.unsqueeze(1), pos.unsqueeze(1),
                                   torch.ones(len(batch_prices), 1)], dim=1)

                action, hidden = model(state, hidden=hidden, portfolio_values=portfolio, prices=p_curr,
                                       prev_positions=pos, transaction_cost=0.01)

                cost = 0.01 * p_curr * torch.abs(action - pos)
                pnl = (action * (p_next - p_curr)) - cost
                portfolio += pnl
                pos = action

            payoff = torch.relu(strike - norm_prices[:, -1, 0])
            error = payoff - portfolio
            mse = torch.mean(error ** 2).item()

            print(f"MSE: {mse:.4f} | RMSE: ${np.sqrt(mse):.2f}")
            results[regime_name] = mse

    return results


if __name__ == "__main__":
    # Locate the model file correctly using utils.paths()
    path_dict = paths()
    model_filename = 'agent_RNNFNN_hybrid.pth'  # Use 'agent_RNNFNN_hybrid.pth' for Experiment 2

    # Construct absolute path
    full_model_path = os.path.join(path_dict['models'], model_filename)

    # Check if file exists before running
    if not os.path.exists(full_model_path):
        print(f"ERROR: Model file not found at {full_model_path}")
        print("Did you run RL.py first?")
    else:
        run_robustness_check(
            model_path=full_model_path,
            network='RNNFNN',
            use_predictor=True  # Change to True if loading hybrid model
        )