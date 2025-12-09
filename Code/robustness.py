# Robustness Check using synthetic data
# Daman Dhaliwal

# import libraries
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

from RL import DeepAgent, bs_price, load_predictors, build_state_features
from ivs_create import create_ivs
from LSTM import train_predictor
from utils import paths


# generate synthetic data
def generate_gbm_path(S0, mu, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
    W = np.cumsum(dW, axis=1)
    t = np.linspace(dt, T, n_steps)

    # GBM Formula
    # Broadcasting S0 and W to shapes
    S_t = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

    # Add S0 at the start
    S_0_col = np.full((n_paths, 1), S0)
    price_paths = np.hstack([S_0_col, S_t])

    return price_paths


# create ivs
def get_representative_ivs_data(target_vol, ivs_data, ivs_flat, ae_encoder, ae_scaler):
    surface_means = np.mean(ivs_flat, axis=1)

    # Find index with closest volatility
    idx = (np.abs(surface_means - target_vol)).argmin()

    # Extract data
    real_surface = ivs_flat[idx]

    # Encode to latent (Agent expects this)
    real_surface_scaled = ae_scaler.transform(real_surface.reshape(1, -1))
    with torch.no_grad():
        latent_vec = ae_encoder(torch.tensor(real_surface_scaled, dtype=torch.float32)).numpy()

    return latent_vec, real_surface_scaled


def run_synthetic_robustness(model_path, network='RNNFNN', state_mode='hybrid', n_episodes=1000):
    path_dict = paths()
    # Load Predictors
    pred_latent, pred_full, _, scaler_full = load_predictors(state_mode, use_predictor=True)

    # Load Autoencoder (for encoding representative surfaces)
    _, ae_scaler, ae_model = train_predictor(n_epochs=0, autoencoder=True, load_autoencoder=True)
    ae_model.eval()
    ae_encoder = ae_model.encoder

    # Load raw IVS data to sample from
    ivs_data, _ = create_ivs(overwrite=False)
    ivs_flat = ivs_data.reshape((ivs_data.shape[0], -1))

    # Load the RL Agent
    feature_dim = 32 + 374 if state_mode == 'hybrid' else 32
    state_dim = 5 + feature_dim

    agent = DeepAgent(network, state_dim, 56, 2, 0.5)

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    agent.load_state_dict(torch.load(model_path))
    agent.eval()

    # Define Regimes
    regimes = {
        'Bull Market': {'mu': 0.15, 'sigma': 0.12},  # Steady growth, low vol
        'Bear Market': {'mu': -0.15, 'sigma': 0.25},  # Slow decline, high vol
        'Crash': {'mu': -0.50, 'sigma': 0.50},  # Massive drop, huge vol
        'Boom': {'mu': 0.40, 'sigma': 0.20},  # Massive rally
    }

    results = []

    # Simulation Loop
    for regime_name, params in regimes.items():
        print(f"Simulating Regime: {regime_name}...")
        T_days = 63

        # FIXED: Variable renamed to 'price_paths' to avoid collision with utils.paths
        price_paths = generate_gbm_path(
            S0=100.0,
            mu=params['mu'],
            sigma=params['sigma'],
            T=T_days / 252,
            n_steps=T_days,
            n_paths=n_episodes
        )

        latent_vec, full_vec = get_representative_ivs_data(
            target_vol=params['sigma'],
            ivs_data=ivs_data,
            ivs_flat=ivs_flat,
            ae_encoder=ae_encoder,
            ae_scaler=ae_scaler
        )

        batch_latent = torch.tensor(latent_vec, dtype=torch.float32).repeat(n_episodes, 1)
        batch_full = torch.tensor(full_vec, dtype=torch.float32).repeat(n_episodes, 1)

        prices = torch.tensor(price_paths, dtype=torch.float32)
        norm_prices = prices  # S0 is 100

        # Initial Portfolio
        strike = 100.0
        # BS Price for initial premium
        init_S = norm_prices[:, 0].numpy()
        init_premium = bs_price(init_S, strike, T_days / 252, 0.0, params['sigma'], True)  # Put

        rl_portfolio = torch.tensor(init_premium, dtype=torch.float32)
        rl_pos = torch.zeros(n_episodes)

        bs_portfolio = torch.tensor(init_premium, dtype=torch.float32)
        bs_pos = np.zeros(n_episodes)

        hidden = None

        for t in range(T_days):
            curr_prices = norm_prices[:, t]

            latent_seq = batch_latent.unsqueeze(1).repeat(1, 5, 1)  # (N, 5, 32)
            full_seq = batch_full.unsqueeze(1).repeat(1, 5, 1)  # (N, 5, 374)

            state_feats = build_state_features(
                state_mode=state_mode,
                use_predictor=True,
                curr_latent=batch_latent,
                curr_full=batch_full,
                predictor_latent=pred_latent,
                predictor_full=pred_full,
                latent_history=latent_seq,
                full_history=full_seq
            )

            time_rem = torch.full((n_episodes, 1), (T_days - t) / T_days)

            state_vec = torch.cat([
                state_feats,
                curr_prices.unsqueeze(1),
                time_rem,
                rl_portfolio.unsqueeze(1),
                rl_pos.unsqueeze(1),
                torch.ones(n_episodes, 1)  # is_put = 1
            ], dim=1)

            action, hidden = agent(
                state_vec, hidden, rl_portfolio, curr_prices, rl_pos, transaction_cost=0.01
            )

            if t < T_days - 1:
                next_prices = norm_prices[:, t + 1]
                trade_size = torch.abs(action - rl_pos)
                cost = 0.01 * curr_prices * trade_size
                pnl = action * (next_prices - curr_prices) - cost
                rl_portfolio += pnl
                rl_pos = action

            # BS Logic
            T_rem_val = (T_days - t) / 252
            if T_rem_val < 1e-5: T_rem_val = 1e-5

            d1 = (np.log(curr_prices.numpy() / strike) + (0.5 * params['sigma'] ** 2) * T_rem_val) / (
                    params['sigma'] * np.sqrt(T_rem_val))
            bs_delta_val = norm.cdf(d1) - 1  # Put Delta

            if t < T_days - 1:
                bs_trade = np.abs(bs_delta_val - bs_pos)
                bs_cost = 0.01 * curr_prices.numpy() * bs_trade
                bs_pnl = bs_delta_val * (next_prices.numpy() - curr_prices.numpy()) - bs_cost
                bs_portfolio += torch.tensor(bs_pnl)
                bs_pos = bs_delta_val

        final_S = norm_prices[:, -1]
        payoff = torch.relu(strike - final_S)

        rl_error = (payoff - rl_portfolio) ** 2
        bs_error = (payoff - bs_portfolio) ** 2

        rl_mse = torch.mean(rl_error).item()
        bs_mse = torch.mean(bs_error).item()

        print(f"   > RL MSE: {rl_mse:.4f} | BS MSE: {bs_mse:.4f}")

        results.append({
            'Regime': regime_name,
            'RL MSE': rl_mse,
            'BS MSE': bs_mse,
            'Imp (%)': (bs_mse - rl_mse) / bs_mse * 100
        })

    # 5. Output Results (LaTeX)
    res_df = pd.DataFrame(results)

    latex_code = res_df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Robustness Check: Hedging Performance across Synthetic Market Regimes",
        label="tab:synthetic_robustness",
        column_format="l|rr|r",
        position="htbp"
    )

    print(latex_code)

    # Save to file
    with open(os.path.join(path_dict['tables'], 'synthetic_robustness.tex'), 'w') as f:
        f.write(latex_code)

    res_df.plot(x='Regime', y=['BS MSE', 'RL MSE'], kind='bar', figsize=(10, 6))
    plt.title('Hedging Error by Market Regime (Synthetic)')
    plt.ylabel('MSE')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(path_dict['plots'], 'synthetic_robustness.png'))
    plt.close()


if __name__ == "__main__":
    path_dict = paths()
    model_name = 'agent_RNNFNN_hybrid_pred_put.pth'
    full_path = os.path.join(path_dict['models'], model_name)

    run_synthetic_robustness(full_path)