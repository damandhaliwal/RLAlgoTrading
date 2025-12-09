# Estimate baseline using Black-Scholes
# Daman Dhaliwal

# import libraries
import numpy as np
import pandas as pd
from scipy.stats import norm

from ivs_create import create_ivs
from spy_prices import get_spy_prices


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


def run_bs_baseline(
        is_put=True,
        transaction_cost=0.01,
        nbs_point_traj=63,
        batch_size=32,
        n_test_batches=100,
        test_split=0.2,
        sigma=0.2
):
    option_type = 'put' if is_put else 'call'
    print(f"\nRunning BS Delta Baseline: {option_type}")
    print("-" * 40)

    # Load data
    ivs_data, ivs_dates = create_ivs(overwrite=False)

    spy_df = get_spy_prices()
    spy_df['date'] = pd.to_datetime(spy_df['date'])

    latent_df = pd.DataFrame({'date': ivs_dates})
    latent_df['latent_idx'] = range(len(latent_df))

    merged_df = pd.merge(spy_df, latent_df, on='date', how='inner').sort_values('date')
    prices = merged_df['spy_price'].values

    # Test split only
    split_idx = int(len(prices) * (1 - test_split))
    test_prices = prices[split_idx:]

    max_start_idx = len(test_prices) - nbs_point_traj
    print(f"Test days: {len(test_prices)}")

    test_losses = []

    for _ in range(n_test_batches):
        # Sample batch
        start_indices = np.random.randint(0, max_start_idx, size=batch_size)

        batch_prices = []
        for start_idx in start_indices:
            batch_prices.append(test_prices[start_idx:start_idx + nbs_point_traj])

        b_prices = np.array(batch_prices)

        # Normalize (S_0 = 100)
        initial_prices = b_prices[:, 0:1]
        scale_factors = 100.0 / initial_prices
        norm_prices = b_prices * scale_factors

        # Initialize
        strike = 100.0
        init_S_norm = norm_prices[:, 0]
        portfolio_value = bs_price(init_S_norm, strike, nbs_point_traj / 252, 0.0, sigma, is_put)
        prev_position = np.zeros(batch_size)

        for t in range(nbs_point_traj - 1):
            curr_prices = norm_prices[:, t]
            next_prices = norm_prices[:, t + 1]

            T_remaining = (nbs_point_traj - t) / 252

            # BS Delta
            delta = bs_delta(curr_prices, strike, T_remaining, 0.0, sigma, is_put)

            # Position: for puts, hold -delta (positive shares); for calls, hold delta
            action = delta

            # Transaction costs
            trade_size = np.abs(action - prev_position)
            costs = transaction_cost * curr_prices * trade_size

            # P&L
            pnl = action * (next_prices - curr_prices) - costs
            portfolio_value = portfolio_value + pnl
            prev_position = action

        # Final payoff
        final_prices = norm_prices[:, -1]
        if is_put:
            payoff = np.maximum(strike - final_prices, 0)
        else:
            payoff = np.maximum(final_prices - strike, 0)

        # Hedging error
        hedging_error = payoff - portfolio_value
        mse = np.mean(hedging_error ** 2)
        test_losses.append(mse)

    avg_mse = np.mean(test_losses)
    std_mse = np.std(test_losses)

    print(f"BS Delta MSE: {avg_mse:.4f} (+/- {std_mse:.4f})")

    return avg_mse


if __name__ == "__main__":
    put_mse = run_bs_baseline(is_put=True)
    call_mse = run_bs_baseline(is_put=False)
    print(f"BS Delta Put MSE:  {put_mse:.4f}")
    print(f"BS Delta Call MSE: {call_mse:.4f}")