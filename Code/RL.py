# Reinforcement Learning for Option Hedging
# Daman Dhaliwal

# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from itertools import product
import pickle

from LSTM import train_predictor
from ivs_create import create_ivs
from spy_prices import get_spy_prices
from utils import paths
from bs_baseline import run_bs_baseline


def bs_price(S, K, T, r, sigma, is_put=True):
    T = np.maximum(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if is_put:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def bs_delta(S, K, T, r, sigma, is_put=True):
    T = np.maximum(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if is_put:
        return norm.cdf(d1) - 1
    else:
        return norm.cdf(d1)


class MarketEnvironment:
    def __init__(self, prices, latents, full_surfaces, n_days=64):
        self.prices = prices
        self.latents = latents
        self.full_surfaces = full_surfaces
        self.n_days = n_days
        self.max_start_idx = len(prices) - n_days

    def get_batch_data(self, batch_size):
        start_indices = np.random.randint(0, self.max_start_idx, size=batch_size)

        batch_prices = []
        batch_latents = []
        batch_full = []

        for start_idx in start_indices:
            end_idx = start_idx + self.n_days
            batch_prices.append(self.prices[start_idx:end_idx])
            batch_latents.append(self.latents[start_idx:end_idx])
            batch_full.append(self.full_surfaces[start_idx:end_idx])

        return (
            torch.tensor(np.array(batch_prices), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(batch_latents), dtype=torch.float32),
            torch.tensor(np.array(batch_full), dtype=torch.float32)
        )


class DeepAgent(nn.Module):
    def __init__(self, network, state_dim, hidden_dim, num_layers, dropout_par, nbs_assets=1, max_borrow=100.0):
        super(DeepAgent, self).__init__()
        self.network = network
        self.max_borrow = max_borrow

        if network == 'RNNFNN':
            self.lstm = nn.LSTM(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout_par
            )
            self.fnn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout_par),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout_par),
                nn.Linear(hidden_dim, nbs_assets)
            )

        elif network == 'LSTM':
            self.lstm = nn.LSTM(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_par
            )
            self.fc = nn.Linear(hidden_dim, nbs_assets)

        elif network == 'FFNN':
            self.ffnn = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout_par),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout_par),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout_par),
                nn.Linear(hidden_dim, nbs_assets)
            )

    def forward(self, x, hidden=None, portfolio_values=None, prices=None, prev_positions=None, transaction_cost=0.0):
        raw_action = None
        new_hidden = None

        if self.network == 'RNNFNN':
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            lstm_out, new_hidden = self.lstm(x, hidden)
            raw_action = self.fnn(lstm_out[:, -1, :]).squeeze(-1)

        elif self.network == 'LSTM':
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            lstm_out, new_hidden = self.lstm(x, hidden)
            raw_action = self.fc(lstm_out[:, -1, :]).squeeze(-1)

        elif self.network == 'FFNN':
            raw_action = self.ffnn(x).squeeze(-1)
            new_hidden = None

        # Leverage Constraint
        if portfolio_values is not None:
            numerator = portfolio_values + self.max_borrow + (transaction_cost * prices * prev_positions)
            denominator = prices * (1 + transaction_cost) + 1e-8
            upper_bound = numerator / denominator
            constrained_action = torch.minimum(raw_action, upper_bound)
            return constrained_action, new_hidden

        return raw_action, new_hidden


def compute_loss(hedging_error, loss_type='MSE', alpha=0.95):
    if loss_type == 'MSE':
        return torch.mean(hedging_error ** 2)
    elif loss_type == 'SMSE':
        return torch.mean(torch.relu(hedging_error) ** 2)
    elif loss_type == 'CVaR':
        cutoff = int(alpha * hedging_error.shape[0])
        sorted_errors, _ = torch.sort(hedging_error)
        return torch.mean(sorted_errors[cutoff:])
    return torch.mean(hedging_error ** 2)


def get_state_dim(state_mode, use_predictor, latent_dim=32, full_dim=374):
    base_dim = 5

    if state_mode == 'latent':
        feature_dim = latent_dim
        if use_predictor:
            feature_dim += latent_dim  # current + predicted latent
    elif state_mode == 'full':
        feature_dim = full_dim
        if use_predictor:
            feature_dim += full_dim  # current + predicted full
    elif state_mode == 'hybrid':
        # Hybrid: 24-dim current + 374-dim predicted
        feature_dim = latent_dim + full_dim
    else:
        raise ValueError(f"Unknown state_mode: {state_mode}")

    return base_dim + feature_dim


def build_state_features(state_mode, use_predictor, curr_latent, curr_full,
                         predictor_latent=None, predictor_full=None, sequence_length=5,
                         latent_history=None, full_history=None):
    features = []

    if state_mode == 'latent':
        features.append(curr_latent)
        if use_predictor and predictor_latent is not None:
            with torch.no_grad():
                predicted = predictor_latent(latent_history)
            features.append(predicted)

    elif state_mode == 'full':
        features.append(curr_full)
        if use_predictor and predictor_full is not None:
            with torch.no_grad():
                predicted = predictor_full(full_history)
            features.append(predicted)

    elif state_mode == 'hybrid':
        # Current latent + predicted full surface
        features.append(curr_latent)
        if predictor_full is not None:
            with torch.no_grad():
                predicted = predictor_full(full_history)
            features.append(predicted)
        else:
            raise ValueError("Hybrid mode requires predictor_full")

    return torch.cat(features, dim=1)


def load_predictors(state_mode, use_predictor):
    path_dict = paths()
    models_dir = path_dict['models']

    predictor_latent = None
    predictor_full = None
    scaler_latent = None
    scaler_full = None

    if use_predictor or state_mode == 'hybrid':
        if state_mode == 'latent':
            # Load latent predictor
            predictor_latent, scaler_latent, _ = train_predictor(
                n_epochs=0, autoencoder=True, load_autoencoder=True
            )
            predictor_latent.eval()

        elif state_mode == 'full':
            # Load full surface predictor
            predictor_full, scaler_full, _ = train_predictor(
                n_epochs=0, autoencoder=False, load_autoencoder=True
            )
            predictor_full.eval()

        elif state_mode == 'hybrid':
            # Load full surface predictor for hybrid
            predictor_full, scaler_full, _ = train_predictor(
                n_epochs=0, autoencoder=False, load_autoencoder=True
            )
            predictor_full.eval()

    return predictor_latent, predictor_full, scaler_latent, scaler_full


def train_hedging_agent(
        # Architecture
        network='RNNFNN',
        hidden_dim=56,
        num_layers=2,
        dropout_par=0.5,
        nbs_assets=1,

        # Training
        n_epochs=50,
        lr=0.001,
        batch_size=32,
        batches_per_epoch=100,

        # Option specification
        strike=100.0,
        is_put=True,

        # Market frictions
        transaction_cost=0.01,

        # Loss
        loss_type='MSE',
        alpha=0.95,

        # Episode
        nbs_point_traj=63,
        sequence_length=5,  # For LSTM predictor

        # State representation: 'latent', 'full', 'hybrid'
        state_mode='latent',
        use_predictor=False,

        # Data loading
        ivs_overwrite=False,
        test_split=0.2,

        # Output
        verbose=True,
        save_model=True
):
    # Validate configuration
    if state_mode == 'hybrid' and not use_predictor:
        raise ValueError("Hybrid mode requires use_predictor=True")

    path_dict = paths()
    option_type = 'put' if is_put else 'call'

    # load autoencode
    _, scaler_ae, autoencoder = train_predictor(
        n_epochs=0, autoencoder=True, load_autoencoder=True
    )
    autoencoder.eval()

    # load predictors
    predictor_latent, predictor_full, _, scaler_full = load_predictors(state_mode, use_predictor)

    # load and process data
    ivs_data, ivs_dates = create_ivs(overwrite=ivs_overwrite)
    ivs_flat = ivs_data.reshape((ivs_data.shape[0], -1))  # (N, 374)

    # Scale for autoencoder
    ivs_scaled_ae = scaler_ae.transform(ivs_flat)

    # Encode to latent
    with torch.no_grad():
        latents_encoded = autoencoder.encoder(
            torch.tensor(ivs_scaled_ae, dtype=torch.float32)
        ).numpy()

    # Scale full surfaces separately for full/hybrid modes
    if state_mode in ['full', 'hybrid']:
        from sklearn.preprocessing import StandardScaler
        scaler_full_surface = StandardScaler()
        full_surfaces_scaled = scaler_full_surface.fit_transform(ivs_flat)
    else:
        full_surfaces_scaled = ivs_scaled_ae  # Use same scaling as AE

    # align prices
    spy_df = get_spy_prices()
    spy_df['date'] = pd.to_datetime(spy_df['date'])

    latent_df = pd.DataFrame({'date': ivs_dates})
    latent_df['latent_idx'] = range(len(latent_df))

    merged_df = pd.merge(spy_df, latent_df, on='date', how='inner').sort_values('date')

    if len(merged_df) == 0:
        raise ValueError("No overlapping dates between prices and IVS data!")

    prices = merged_df['spy_price'].values
    valid_indices = merged_df['latent_idx'].values
    latents_aligned = latents_encoded[valid_indices]
    full_surfaces_aligned = full_surfaces_scaled[valid_indices]

    # === Train/Test Split ===
    split_idx = int(len(prices) * (1 - test_split))

    train_env = MarketEnvironment(
        prices[:split_idx],
        latents_aligned[:split_idx],
        full_surfaces_aligned[:split_idx],
        n_days=nbs_point_traj
    )

    test_env = MarketEnvironment(
        prices[split_idx:],
        latents_aligned[split_idx:],
        full_surfaces_aligned[split_idx:],
        n_days=nbs_point_traj
    )

    # initialize agent
    state_dim = get_state_dim(state_mode, use_predictor)

    model = DeepAgent(network, state_dim, hidden_dim, num_layers, dropout_par, nbs_assets)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    model.train()
    is_put_val = 1.0 if is_put else 0.0
    train_losses = []

    for epoch in range(n_epochs):
        epoch_losses = []

        for _ in range(batches_per_epoch):
            b_prices, b_latents, b_full = train_env.get_batch_data(batch_size)

            # Normalize prices (S_0 = 100)
            initial_prices = b_prices[:, 0, 0].unsqueeze(1)
            scale_factors = 100.0 / initial_prices
            norm_prices = b_prices * scale_factors.unsqueeze(1)

            # Initialize portfolio
            current_strike = 100.0
            init_S_norm = norm_prices[:, 0, 0].numpy()
            init_premium = bs_price(init_S_norm, current_strike, nbs_point_traj / 252, 0.0, 0.2, is_put)

            portfolio_value = torch.tensor(init_premium, dtype=torch.float32)
            prev_position = torch.zeros(batch_size)
            hidden_state = None

            for t in range(sequence_length, nbs_point_traj - 1):
                curr_prices = norm_prices[:, t, 0]
                next_prices = norm_prices[:, t + 1, 0]
                curr_latent = b_latents[:, t, :]
                curr_full = b_full[:, t, :]

                # Get history for LSTM predictor
                latent_history = b_latents[:, t - sequence_length:t, :]
                full_history = b_full[:, t - sequence_length:t, :]

                # Build state features
                state_features = build_state_features(
                    state_mode=state_mode,
                    use_predictor=use_predictor,
                    curr_latent=curr_latent,
                    curr_full=curr_full,
                    predictor_latent=predictor_latent,
                    predictor_full=predictor_full,
                    latent_history=latent_history,
                    full_history=full_history
                )

                time_rem = torch.full((batch_size, 1), (nbs_point_traj - t) / nbs_point_traj)

                state_vec = torch.cat([
                    state_features,
                    curr_prices.unsqueeze(1),
                    time_rem,
                    portfolio_value.unsqueeze(1),
                    prev_position.unsqueeze(1),
                    torch.full((batch_size, 1), is_put_val)
                ], dim=1)

                action, hidden_state = model(
                    state_vec,
                    hidden=hidden_state,
                    portfolio_values=portfolio_value,
                    prices=curr_prices,
                    prev_positions=prev_position,
                    transaction_cost=transaction_cost
                )

                trade_size = torch.abs(action - prev_position)
                costs = transaction_cost * curr_prices * trade_size
                pnl = (action * (next_prices - curr_prices)) - costs
                portfolio_value = portfolio_value + pnl
                prev_position = action

            # Final payoff
            final_prices = norm_prices[:, -1, 0]
            if is_put:
                payoff = torch.relu(current_strike - final_prices)
            else:
                payoff = torch.relu(final_prices - current_strike)

            hedging_error = payoff - portfolio_value
            loss = compute_loss(hedging_error, loss_type, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")

    # Testing
    model.eval()
    test_losses = []

    with torch.no_grad():
        for _ in range(100):
            b_prices, b_latents, b_full = test_env.get_batch_data(batch_size)

            initial_prices = b_prices[:, 0, 0].unsqueeze(1)
            scale_factors = 100.0 / initial_prices
            norm_prices = b_prices * scale_factors.unsqueeze(1)

            current_strike = 100.0
            init_S_norm = norm_prices[:, 0, 0].numpy()
            init_premium = bs_price(init_S_norm, current_strike, nbs_point_traj / 252, 0.0, 0.2, is_put)

            portfolio_value = torch.tensor(init_premium, dtype=torch.float32)
            prev_position = torch.zeros(batch_size)
            hidden_state = None

            for t in range(sequence_length, nbs_point_traj - 1):
                curr_prices = norm_prices[:, t, 0]
                next_prices = norm_prices[:, t + 1, 0]
                curr_latent = b_latents[:, t, :]
                curr_full = b_full[:, t, :]

                latent_history = b_latents[:, t - sequence_length:t, :]
                full_history = b_full[:, t - sequence_length:t, :]

                state_features = build_state_features(
                    state_mode=state_mode,
                    use_predictor=use_predictor,
                    curr_latent=curr_latent,
                    curr_full=curr_full,
                    predictor_latent=predictor_latent,
                    predictor_full=predictor_full,
                    latent_history=latent_history,
                    full_history=full_history
                )

                time_rem = torch.full((batch_size, 1), (nbs_point_traj - t) / nbs_point_traj)

                state_vec = torch.cat([
                    state_features,
                    curr_prices.unsqueeze(1),
                    time_rem,
                    portfolio_value.unsqueeze(1),
                    prev_position.unsqueeze(1),
                    torch.full((batch_size, 1), is_put_val)
                ], dim=1)

                action, hidden_state = model(
                    state_vec,
                    hidden=hidden_state,
                    portfolio_values=portfolio_value,
                    prices=curr_prices,
                    prev_positions=prev_position,
                    transaction_cost=transaction_cost
                )

                trade_size = torch.abs(action - prev_position)
                costs = transaction_cost * curr_prices * trade_size
                pnl = (action * (next_prices - curr_prices)) - costs
                portfolio_value = portfolio_value + pnl
                prev_position = action

            final_prices = norm_prices[:, -1, 0]
            if is_put:
                payoff = torch.relu(current_strike - final_prices)
            else:
                payoff = torch.relu(final_prices - current_strike)

            hedging_error = payoff - portfolio_value
            test_loss = compute_loss(hedging_error, loss_type, alpha)
            test_losses.append(test_loss.item())

    avg_test_loss = np.mean(test_losses)

    print(f"Final Test Loss: {avg_test_loss:.6f}")

    # === Save Model ===
    if save_model:
        pred_str = "pred" if use_predictor else "nopred"
        model_name = f"agent_{network}_{state_mode}_{pred_str}_{option_type}.pth"
        save_path = os.path.join(path_dict['models'], model_name)
        torch.save(model.state_dict(), save_path)

    results = {
        'state_mode': state_mode,
        'use_predictor': use_predictor,
        'is_put': is_put,
        'train_loss_final': train_losses[-1] if train_losses else None,
        'test_loss': avg_test_loss,
        'state_dim': state_dim
    }

    return model, results


def run_experiment_grid(n_epochs=50, network='RNNFNN'):
    configurations = [
        # (state_mode, use_predictor)
        ('latent', False),
        ('latent', True),
        ('full', False),
        ('full', True),
        ('hybrid', True),  # Hybrid requires predictor
    ]

    option_types = [True, False]  # is_put

    all_results = []

    for (state_mode, use_predictor), is_put in product(configurations, option_types):
        try:
            _, results = train_hedging_agent(
                network=network,
                n_epochs=n_epochs,
                state_mode=state_mode,
                use_predictor=use_predictor,
                is_put=is_put,
                verbose=True
            )
            all_results.append(results)

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                'state_mode': state_mode,
                'use_predictor': use_predictor,
                'is_put': is_put,
                'error': str(e)
            })

    results_df = pd.DataFrame(all_results)

    # Save results
    path_dict = paths()
    results_df.to_csv(os.path.join(path_dict['tables'], 'experiment_results.csv'), index=False)

    print(results_df.to_string())

    return results_df

# present results
def generate_final_table(n_epochs = 100):
    # Get Black-Scholes Baselines
    bs_put_mse = run_bs_baseline(is_put=True)
    bs_call_mse = run_bs_baseline(is_put=False)

    baselines = {
        'Put': bs_put_mse,
        'Call': bs_call_mse
    }

    # This runs: Latent(NoPred), Latent(Pred), Full(NoPred), Full(Pred), Hybrid(Pred)
    # for both Puts and Calls.
    rl_results_df = run_experiment_grid(n_epochs = n_epochs, network = 'RNNFNN')

    strategies = [
        ('BS Delta (Baseline)', None, None),
        ('RL-Latent (No Pred)', 'latent', False),
        ('RL-Latent (Pred)', 'latent', True),
        ('RL-Full (No Pred)', 'full', False),
        ('RL-Full (Pred)', 'full', True),
        ('RL-Hybrid (Proposed)', 'hybrid', True)
    ]

    final_rows = []

    for label, mode, use_pred in strategies:
        row_data = {'Strategy': label}

        for opt_type, is_put in [('Call', False), ('Put', True)]:
            if label == 'BS Delta (Baseline)':
                mse = baselines[opt_type]
            else:
                mask = (rl_results_df['state_mode'] == mode) & \
                       (rl_results_df['use_predictor'] == use_pred) & \
                       (rl_results_df['is_put'] == is_put)

                if mask.sum() > 0:
                    mse = rl_results_df.loc[mask, 'test_loss'].values[0]
                else:
                    mse = np.nan

            # Calculate Metrics
            rmse = np.sqrt(mse)
            bs_ref = baselines[opt_type]
            improvement = ((bs_ref - mse) / bs_ref) * 100

            # Add to row
            row_data[f'{opt_type} MSE'] = mse
            row_data[f'{opt_type} RMSE'] = rmse
            row_data[f'{opt_type} Imp (%)'] = improvement

        final_rows.append(row_data)

    # Create DataFrame
    final_table = pd.DataFrame(final_rows)

    # Reorder columns for readability
    cols = ['Strategy',
            'Put MSE', 'Put RMSE', 'Put Imp (%)',
            'Call MSE', 'Call RMSE', 'Call Imp (%)']
    final_table = final_table[cols]

    latex_code = final_table.to_latex(
        index=False,
        float_format="%.4f",
        caption="Comparative Hedging Performance (MSE & RMSE) across 63-day Test Horizon",
        label="tab:hedging_results",
        column_format="l|rrr|rrr",  # Vertical bar separates Strategy from Data
        position="H"
    )

    print(latex_code)

    path = paths()
    with open(os.path.join(path['tables'], 'hedging_results.tex'), 'w') as f:
        f.write(latex_code)


if __name__ == "__main__":
    generate_final_table(n_epochs = 100)