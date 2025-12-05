# Deep Hedging with Predictive Market Dynamics

This repository contains a Deep Reinforcement Learning (DRL) framework for optimizing options hedging strategies. It extends the state-of-the-art "Deep Hedging" literature by integrating a **predictive module for the Implied Volatility Surface (IVS)** directly into the agent's decision process.

## 🚀 Project Overview

  * **Objective:** Minimize hedging error for SPY options portfolios under transaction costs and market friction.
  * **Core Innovation:** Traditional Deep Hedging agents react to current market states. This system incorporates a **Forecasting Module** (LSTM) that predicts the evolution of the Implied Volatility Surface, allowing the agent to preemptively adjust hedge ratios based on anticipated market shifts.
  * **Methodology:**
    1.  **Dimensionality Reduction:** Compressed high-dimensional IVS data (374 points) into a compact latent representation using an **Autoencoder**.
    2.  **Temporal Modeling:** Trained an **LSTM** to model the dynamics of these latent features over time.
    3.  **Policy Optimization:** Trained a Recurrent Reinforcement Learning agent (RNN-FNN) to output optimal hedge ratios, conditioning on both realized prices and predicted volatility dynamics.
    4.  **Robustness Testing:** Evaluated agent performance across distinct market regimes (Bull, Bear, Crisis) to ensure stability under stress.

## 🏗 System Architecture

The pipeline consists of three coupled neural networks implemented in **PyTorch**:

### 1\. Latent State Representation (Autoencoder)

  * **Input:** Normalized Implied Volatility Surface (Maturity $\times$ Delta).
  * **Architecture:** Symmetric Encoder-Decoder with dense layers and ReLU activations.
  * **Purpose:** Reduces the 374-dimensional surface to a robust 32-dimensional latent vector, removing noise and extracting the principal factors of volatility (Level, Slope, Curvature).

### 2\. Dynamics Forecasting (LSTM Predictor)

  * **Input:** Sequence of past latent vectors ($t-k, ..., t$).
  * **Output:** Predicted latent vector for $t+1$.
  * **Role:** Acts as a "World Model," providing the RL agent with foresight into how the volatility surface—and thus option pricing—will evolve.

### 3\. Hedging Policy (Deep RL Agent)

  * **Type:** Policy Gradient / Direct Policy Search (Deep Hedging).
  * **Architecture:** Hybrid **RNN-FNN**.
      * *LSTM Block:* Processes the history of market states and predictions to capture non-Markovian dependencies.
      * *FNN Block:* Maps the recurrent embedding to constrained actions (hedge ratios).
  * **Constraints:** Enforces leverage constraints and accounts for proportional transaction costs ($1\%$) in the reward function.

## 📂 Repository Structure

```bash
├── Code/
│   ├── RL.py                 # Core DeepAgent implementation and Training Loop (bootstrapped simulation)
│   ├── LSTM.py               # Dynamics model: LSTM predictor for IVS latent representations
│   ├── autoencoder.py        # Dimensionality reduction for the Implied Volatility Surface
│   ├── robustness.py         # Regime detection (Crisis/Bull/Bear) and stress testing
│   ├── ivs_create.py         # Data engineering: Constructing 3D IVS from raw options data
│   ├── spy_prices.py         # Financial data ingestion (Yahoo Finance API)
│   ├── plots.py              # Visualization of 3D Volatility Surfaces
│   └── utils.py              # Path management and configuration
├── Text/                     # Research abstract and documentation
└── README.md
```

## 📊 Evaluation & Robustness

To validate the strategy for real-world deployment, the agent was stress-tested against historical market regimes identified via volatility clustering:

  * **Regime Classification:** Automatically segments history into *Normal*, *Bull*, *Bear*, and *Crisis* periods based on realized volatility and return thresholds.
  * **Performance Metrics:**
      * **MSE (Mean Squared Hedging Error):** Primary loss function.
      * **CVaR (Conditional Value at Risk):** Tail risk assessment.
  * **Results:** The hybrid Agent (with Predictor) demonstrated superior stability in high-volatility "Crisis" regimes compared to the baseline reacting solely to current states.

## 💻 Usage

1.  **Environment Setup:**

    ```bash
    pip install torch pandas numpy scipy scikit-learn matplotlib yfinance
    ```

2.  **Data Pipeline:**
    Ingest SPY data and construct the volatility surfaces.

    ```bash
    python Code/spy_prices.py
    python Code/ivs_create.py
    ```

3.  **Pre-training:**
    Train the Autoencoder and LSTM Predictor to learn market dynamics.

    ```bash
    python Code/autoencoder.py
    python Code/LSTM.py
    ```

4.  **Agent Training:**
    Train the Hedging Agent. Use `use_predictor=True` to enable the forecasting module.

    ```python
    # Inside Code/RL.py
    train_hedging_agent(network='RNNFNN', use_predictor=True)
    ```

5.  **Robustness Check:**
    Evaluate the trained model across different market regimes.

    ```bash
    python Code/robustness.py
    ```

-----

*Author: Damanveer Singh Dhaliwal*
