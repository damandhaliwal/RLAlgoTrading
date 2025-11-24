import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from scipy.stats import norm

from LSTM import train_predictor
from ivs_create import create_ivs
from spy_prices import get_spy_prices
from utils import paths


# ---------------------------------------------------------
# 1. Financial Helpers (Black-Scholes)
# ---------------------------------------------------------
def bs_price(S, K, T, r, sigma, is_put=True):
    """Calculates Black-Scholes price to initialize portfolio wealth."""
    # Avoid div by zero
    T = np.maximum(T, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if is_put:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


# ---------------------------------------------------------
# 2. Market Environment (Handles Data Bootstrapping)
# ---------------------------------------------------------
class MarketEnvironment:
    def __init__(self, prices, latents, n_days=64):
        self.prices = prices
        self.latents = latents
        self.n_days = n_days
        self.max_start_idx = len(prices) - n_days

    def get_batch_data(self, batch_size):
        """Generates a batch of bootstrapped episodes."""
        start_indices = np.random.randint(0, self.max_start_idx, size=batch_size)

        batch_prices = []
        batch_latents = []

        for start_idx in start_indices:
            end_idx = start_idx + self.n_days
            batch_prices.append(self.prices[start_idx:end_idx])
            batch_latents.append(self.latents[start_idx:end_idx])

        # Returns tensors: [batch, time, 1] and [batch, time, latent_dim]
        return (torch.tensor(np.array(batch_prices), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(np.array(batch_latents), dtype=torch.float32))


# ---------------------------------------------------------
# 3. Deep Agent (With Leverage & Memory)
# ---------------------------------------------------------
class DeepAgent(nn.Module):
    def __init__(self, network, state_dim, hidden_dim, num_layers, dropout_par, nbs_assets=1, max_borrow=100.0):
        super(DeepAgent, self).__init__()
        self.network = network
        self.max_borrow = max_borrow

        if network == 'RNNFNN':
            # Paper Architecture: 2 LSTM layers + 2 Feedforward layers
            self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True,
                                dropout=dropout_par)
            self.fnn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout_par),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout_par),
                nn.Linear(hidden_dim, nbs_assets)
            )

        elif network == 'LSTM':
            self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                                dropout=dropout_par)
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
            if len(x.shape) == 2: x = x.unsqueeze(1)
            # Pass hidden state to maintain memory
            lstm_out, new_hidden = self.lstm(x, hidden)
            raw_action = self.fnn(lstm_out[:, -1, :]).squeeze(-1)

        elif self.network == 'LSTM':
            if len(x.shape) == 2: x = x.unsqueeze(1)
            lstm_out, new_hidden = self.lstm(x, hidden)
            raw_action = self.fc(lstm_out[:, -1, :]).squeeze(-1)

        elif self.network == 'FFNN':
            raw_action = self.ffnn(x).squeeze(-1)
            new_hidden = None

            # Leverage Constraint (Paper Eq 6)
        if portfolio_values is not None:
            numerator = portfolio_values + self.max_borrow + (transaction_cost * prices * prev_positions)
            denominator = prices * (1 + transaction_cost) + 1e-8
            upper_bound = numerator / denominator

            # Constrain output
            constrained_action = torch.minimum(raw_action, upper_bound)
            return constrained_action, new_hidden

        return raw_action, new_hidden


# ---------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 5. Main Training Loop
# ---------------------------------------------------------
def train_hedging_agent(
        network='RNNFNN',
        hidden_dim=56,
        num_layers=2,
        dropout_par=0.5,
        nbs_assets=1,
        n_epochs=50,
        lr=0.001,
        batch_size=32,
        strike=100.0,  # Will be relative to normalized price (100)
        is_put=True,
        transaction_cost=0.01,
        loss_type='MSE',
        alpha=0.95,
        nbs_point_traj=63,
        use_autoencoder=True,
        use_predictor=False,
        load_autoencoder=True,
        ivs_overwrite=False,
        test_split=0.2
):
    print(f"--- Starting Training ---\nMode: {'Hybrid (Predictor)' if use_predictor else 'End-to-End (Novelty)'}")

    # 1. Load Models & Data
    predictor = None
    if use_predictor:
        print("Loading LSTM Predictor...")
        predictor, scaler_pred, autoencoder = train_predictor(
            autoencoder=use_autoencoder, load_autoencoder=load_autoencoder, n_epochs=0
        )
        predictor.eval()
    else:
        # Load just AE
        _, scaler_pred, autoencoder = train_predictor(
            n_epochs=0, autoencoder=use_autoencoder, load_autoencoder=load_autoencoder
        )

    if autoencoder: autoencoder.eval()

    # Load IVS (Unpacking Tuple)
    ivs_data, ivs_dates = create_ivs(overwrite=ivs_overwrite)
    ivs_flat = ivs_data.reshape((ivs_data.shape[0], -1))
    ivs_scaled = scaler_pred.transform(ivs_flat)

    # Encode to Latents
    if autoencoder is not None:
        with torch.no_grad():
            latents_encoded = autoencoder.encoder(torch.tensor(ivs_scaled, dtype=torch.float32)).numpy()
    else:
        latents_encoded = ivs_scaled

    # Load Prices
    spy_df = get_spy_prices()
    spy_df['date'] = pd.to_datetime(spy_df['date'])

    # --- ROBUST ALIGNMENT ---
    # Merge Price and Volatility on Date to ensure alignment
    latent_df = pd.DataFrame({'date': ivs_dates})
    latent_df['latent_idx'] = range(len(latent_df))

    merged_df = pd.merge(spy_df, latent_df, on='date', how='inner').sort_values('date')

    if len(merged_df) == 0:
        raise ValueError("Error: No overlapping dates found between Prices and IVS data!")

    prices = merged_df['spy_price'].values
    valid_indices = merged_df['latent_idx'].values
    latents_encoded = latents_encoded[valid_indices]

    print(f"Aligned Data: {len(prices)} common days found.")

    # --- TRAIN / TEST SPLIT ---
    split_idx = int(len(prices) * (1 - test_split))

    train_prices = prices[:split_idx]
    train_latents = latents_encoded[:split_idx]

    test_prices = prices[split_idx:]
    test_latents = latents_encoded[split_idx:]

    print(f"Data Split: {len(train_prices)} training days | {len(test_prices)} testing days")

    # Create Environments
    train_env = MarketEnvironment(train_prices, train_latents, n_days=nbs_point_traj)
    test_env = MarketEnvironment(test_prices, test_latents, n_days=nbs_point_traj)

    # Setup Agent
    feature_dim = 24
    actual_state_dim = 5 + feature_dim

    print(f"Network: {network} | Input Dim: {actual_state_dim}")
    model = DeepAgent(network, actual_state_dim, hidden_dim, num_layers, dropout_par, nbs_assets)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training Loop ---
    model.train()
    is_put_val = 1.0 if is_put else 0.0
    batches_per_epoch = 100

    for epoch in range(n_epochs):
        epoch_losses = []

        for _ in range(batches_per_epoch):
            # 1. Get Batch
            b_prices, b_latents = train_env.get_batch_data(batch_size)

            # --- NORMALIZATION (The Fix for Test Loss) ---
            # Scale everything so S_0 = 100
            initial_prices = b_prices[:, 0, 0].unsqueeze(1)
            scale_factors = 100.0 / initial_prices
            norm_prices = b_prices * scale_factors.unsqueeze(1)
            # ---------------------------------------------

            # Initialize Portfolio (ATM Strike = 100)
            current_strike = 100.0
            init_S_norm = norm_prices[:, 0, 0].numpy()

            # Calculate Option Premium (Cash received)
            init_prices = bs_price(init_S_norm, current_strike, nbs_point_traj / 252, 0.0, 0.2, is_put)

            portfolio_value = torch.tensor(init_prices, dtype=torch.float32)
            prev_position = torch.zeros(batch_size)
            hidden_state = None

            for t in range(nbs_point_traj - 1):
                # Use Normalized Prices
                curr_prices = norm_prices[:, t, 0]
                next_prices = norm_prices[:, t + 1, 0]
                curr_latent = b_latents[:, t, :]

                if use_predictor:
                    with torch.no_grad():
                        feature_input = predictor(curr_latent.unsqueeze(1))
                else:
                    feature_input = curr_latent

                time_rem = torch.full((batch_size, 1), (nbs_point_traj - t) / nbs_point_traj)

                state_vec = torch.cat([
                    feature_input,
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
                portfolio_value += pnl
                prev_position = action

            # Final Payoff (Strike is 100)
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

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss ({loss_type}): {np.mean(epoch_losses):.6f}")

    # --- Evaluation Loop ---
    print("\n--- Running Evaluation on Unseen Test Data ---")
    model.eval()
    test_losses = []

    with torch.no_grad():
        for _ in range(100):
            # USE TEST ENV
            b_prices, b_latents = test_env.get_batch_data(batch_size)

            # --- NORMALIZATION ---
            initial_prices = b_prices[:, 0, 0].unsqueeze(1)
            scale_factors = 100.0 / initial_prices
            norm_prices = b_prices * scale_factors.unsqueeze(1)
            # ---------------------

            current_strike = 100.0
            init_S_norm = norm_prices[:, 0, 0].numpy()
            init_prices = bs_price(init_S_norm, current_strike, nbs_point_traj / 252, 0.0, 0.2, is_put)

            portfolio_value = torch.tensor(init_prices, dtype=torch.float32)
            prev_position = torch.zeros(batch_size)
            hidden_state = None

            for t in range(nbs_point_traj - 1):
                curr_prices = norm_prices[:, t, 0]
                next_prices = norm_prices[:, t + 1, 0]
                curr_latent = b_latents[:, t, :]

                if use_predictor:
                    feature_input = predictor(curr_latent.unsqueeze(1))
                else:
                    feature_input = curr_latent

                time_rem = torch.full((batch_size, 1), (nbs_point_traj - t) / nbs_point_traj)
                state_vec = torch.cat([feature_input, curr_prices.unsqueeze(1), time_rem, portfolio_value.unsqueeze(1),
                                       prev_position.unsqueeze(1), torch.full((batch_size, 1), is_put_val)], dim=1)

                action, hidden_state = model(state_vec, hidden=hidden_state, portfolio_values=portfolio_value,
                                             prices=curr_prices, prev_positions=prev_position,
                                             transaction_cost=transaction_cost)

                trade_size = torch.abs(action - prev_position)
                costs = transaction_cost * curr_prices * trade_size
                pnl = (action * (next_prices - curr_prices)) - costs
                portfolio_value += pnl
                prev_position = action

            final_prices = norm_prices[:, -1, 0]
            if is_put:
                payoff = torch.relu(current_strike - final_prices)
            else:
                payoff = torch.relu(final_prices - current_strike)

            hedging_error = payoff - portfolio_value
            test_loss = compute_loss(hedging_error, loss_type, alpha)
            test_losses.append(test_loss.item())

    print(f"Final Test Loss ({loss_type}): {np.mean(test_losses):.6f}")

    path_dict = paths()
    mode_str = "hybrid" if use_predictor else "e2e"
    torch.save(model.state_dict(), os.path.join(path_dict['models'], f'agent_{network}_{mode_str}.pth'))
    return model


if __name__ == "__main__":
    # Experiment 1: Novelty (End-to-End)
    print(">>> Running Experiment 1: End-to-End (Novelty) <<<")
    model_e2e = train_hedging_agent(
        network='RNNFNN',
        n_epochs=50,
        use_predictor=False,
        loss_type='MSE'
    )

    # Experiment 2: Comparison (Hybrid)
    # Uncomment below to run
    print("\n>>> Running Experiment 2: Hybrid (Comparison) <<<")
    model_hybrid = train_hedging_agent(
         network='RNNFNN',
         n_epochs=50,
         use_predictor=True,
         loss_type='MSE'
     )