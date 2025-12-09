# LSTM Predictor
# Daman Dhaliwal

# import libraries
import os
import pickle

from ivs_create import create_ivs
from utils import paths, load_data
from autoencoder import Autoencoder

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


class IVS_Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(IVS_Predictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        predicted = self.fc(last_output)
        return predicted


def _load_autoencoder():
    path_dict = paths()
    models_dir = path_dict['models']

    with open(os.path.join(models_dir, 'autoencoder_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    data, _ = create_ivs(overwrite=False)
    input_dim = data.reshape((data.shape[0], -1)).shape[1]

    model = Autoencoder(input_dim=input_dim, latent_dim=32)
    model.load_state_dict(torch.load(os.path.join(models_dir, 'autoencoder.pth')))
    model.eval()

    return model, scaler


def create_sequences(data, sequence_length):
    num_sequences = len(data) - sequence_length
    feature_dim = data.shape[1]

    X = np.zeros((num_sequences, sequence_length, feature_dim))
    y = np.zeros((num_sequences, feature_dim))

    for i in range(num_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = data[i + sequence_length]

    return X, y


def train_predictor(
        sequence_length=5,
        hidden_dim=32,
        num_layers=2,
        n_epochs=200,
        overwrite=False,
        autoencoder=True,
        load_autoencoder=True
):
    path_dict = paths()
    models_dir = path_dict['models']
    os.makedirs(models_dir, exist_ok=True)

    if autoencoder:
        model_name = 'lstm_predictor_ae.pth'
        scaler_name = 'lstm_predictor_scaler_ae.pkl'
        mode_str = "latent (32-dim)"
    else:
        model_name = 'lstm_predictor_full.pth'
        scaler_name = 'lstm_predictor_scaler_full.pkl'
        mode_str = "full surface (374-dim)"

    # === Load Data ===
    data, _ = create_ivs(overwrite=overwrite)
    data_flat = data.reshape((data.shape[0], -1))

    # === Setup based on mode ===
    ae_model = None

    if autoencoder:
        ae_model, scaler = _load_autoencoder()

        data_scaled = scaler.transform(data_flat)
        with torch.no_grad():
            predictor_data = ae_model.encoder(
                torch.tensor(data_scaled, dtype=torch.float32)
            ).numpy()
    else:
        scaler = StandardScaler()
        predictor_data = scaler.fit_transform(data_flat)

    input_dim = predictor_data.shape[1]

    print(f"Predictor mode: {mode_str}")
    print(f"Input dimension: {input_dim}")

    if n_epochs == 0:
        model_path = os.path.join(models_dir, model_name)
        scaler_path = os.path.join(models_dir, scaler_name)

        if os.path.exists(model_path):
            print(f"Loading existing predictor: {model_name}")

            predictor = IVS_Predictor(input_dim, hidden_dim, num_layers)
            predictor.load_state_dict(torch.load(model_path))
            predictor.eval()

            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)

            if autoencoder:
                return predictor, scaler, ae_model
            else:
                return predictor, scaler, None
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    X, y = create_sequences(predictor_data, sequence_length)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training sequences: {len(X_train)}, Test sequences: {len(X_test)}")

    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=False
    )

    # initialize model and optimizer
    predictor = IVS_Predictor(input_dim, hidden_dim, num_layers)
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # training loop
    for epoch in range(n_epochs):
        predictor.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predicted = predictor(batch_X)
            loss = loss_fn(predicted, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        predictor.eval()
        test_losses = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predicted = predictor(batch_X)
                loss = loss_fn(predicted, batch_y)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

    # === Save ===
    torch.save(predictor.state_dict(), os.path.join(models_dir, model_name))
    with open(os.path.join(models_dir, scaler_name), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Saved: {model_name}, {scaler_name}")

    if autoencoder:
        return predictor, scaler, ae_model
    else:
        return predictor, scaler, None


def train_both_predictors(n_epochs=200):
    train_predictor(n_epochs=n_epochs, autoencoder=True, load_autoencoder=True)
    train_predictor(n_epochs=n_epochs, autoencoder=False)


# present results
def lstm_results_full(sample_idx = 10):
    path_dict = paths()

    # Load Data & Predictor
    data, dates = create_ivs(overwrite=False)
    data_flat = data.reshape((data.shape[0], -1))

    predictor, scaler, _ = train_predictor(n_epochs=0, autoencoder=False)
    predictor.eval()

    data_scaled = scaler.transform(data_flat)
    sequence_length = 5
    X, y = create_sequences(data_scaled, sequence_length)

    split_idx = int(0.8 * len(X))
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    test_dates = dates[sequence_length:][split_idx:]

    # Predict & Unscale
    inputs = torch.tensor(X_test, dtype=torch.float32)
    targets_scaled = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        preds_scaled = predictor(inputs)

    preds_real = scaler.inverse_transform(preds_scaled.numpy())
    actuals_real = scaler.inverse_transform(targets_scaled.numpy())

    # Calculate Metrics
    mse_real = np.mean((preds_real - actuals_real) ** 2)
    rmse_real = np.sqrt(mse_real)

    naive_preds = actuals_real[:-1]
    naive_targets = actuals_real[1:]
    naive_rmse = np.sqrt(np.mean((naive_preds - naive_targets) ** 2))

    # Get Real Axis Labels
    df = load_data()
    df = df[(((df['delta'] < 0) & (df['cp_flag'] == "P")) |
             ((df['delta'] > 0) & (df['cp_flag'] == "C")))]
    completeness = df.groupby('date').size()
    valid_dates = completeness[completeness == 374].index
    df = df[df['date'].isin(valid_dates)]

    unique_deltas = sorted(df['delta'].unique())
    unique_days = sorted(df['days'].unique())

    # Plot 1: ATM Volatility Time Series
    mid_idx = data_flat.shape[1] // 2
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, actuals_real[:, mid_idx], label='Actual', color='black', alpha=0.6)
    plt.plot(test_dates, preds_real[:, mid_idx], label='Predicted', color='blue', linestyle='--', alpha=0.8)
    plt.title('Full Surface Forecast: Representative ATM Volatility')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(path_dict['plots'], 'figure5.png'), dpi=600)
    plt.close()

    # Plot 2: 3D Surface Comparison
    target_idx = sample_idx
    n_deltas, n_days = data.shape[1], data.shape[2]

    pred_surface = preds_real[target_idx].reshape(n_deltas, n_days)
    actual_surface = actuals_real[target_idx].reshape(n_deltas, n_days)
    current_date = test_dates[target_idx]

    X, Y = np.meshgrid(unique_days, unique_deltas)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, actual_surface, cmap=cm.viridis, edgecolor='none')
    ax1.set_title(f'Actual ({current_date})')
    ax1.set_xlabel('Maturity (Days)')
    ax1.set_ylabel('Delta')
    ax1.set_zlabel('Vol')
    ax1.invert_xaxis()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, pred_surface, cmap=cm.coolwarm, edgecolor='none')
    ax2.set_title(f'LSTM Forecast (t+1)')
    ax2.set_xlabel('Maturity (Days)')
    ax2.set_ylabel('Delta')
    ax2.set_zlabel('Vol')
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(path_dict['plots'], 'lstm_full_surface_comparison.png'), dpi=600)
    plt.close()

    return rmse_real, naive_rmse

def lstm_results_latent(feature_idx=0):
    path_dict = paths()

    # Load Data & Predictor (Latent Mode)
    data, dates = create_ivs(overwrite=False)
    data_flat = data.reshape((data.shape[0], -1))

    # train_predictor with autoencoder=True loads AE model and AE scaler automatically
    predictor, scaler, ae_model = train_predictor(n_epochs=0, autoencoder=True, load_autoencoder=True)
    predictor.eval()
    ae_model.eval()

    # Generate Ground Truth Latents
    data_scaled = scaler.transform(data_flat)
    with torch.no_grad():
        latents_encoded = ae_model.encoder(torch.tensor(data_scaled, dtype=torch.float32)).numpy()

    # Create Sequences
    sequence_length = 5
    X, y = create_sequences(latents_encoded, sequence_length)

    split_idx = int(0.8 * len(X))
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    test_dates = dates[sequence_length:][split_idx:]

    # Predict
    inputs = torch.tensor(X_test, dtype=torch.float32)
    targets = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        preds = predictor(inputs)

    # Metrics
    mse = np.mean((preds.numpy() - targets.numpy()) ** 2)
    rmse = np.sqrt(mse)

    naive_preds = targets.numpy()[:-1]
    naive_targets = targets.numpy()[1:]
    naive_rmse = np.sqrt(np.mean((naive_preds - naive_targets) ** 2))

    # Plot Latent Time Series
    actual_series = targets.numpy()[:, feature_idx]
    pred_series = preds.numpy()[:, feature_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_series, label='Actual Latent', color='black', alpha=0.7)
    plt.plot(test_dates, pred_series, label='Predicted (LSTM)', color='#d62728', linestyle='--', alpha=0.9)

    plt.title(f'Latent Feature #{feature_idx} Forecast (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Latent Value (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(path_dict['plots'], 'lstm_latent_results.png'), dpi=600)
    plt.close()

    return rmse, naive_rmse

if __name__ == "__main__":
    rmse_full, naive_full = lstm_results_full(sample_idx=10)
    print(f"RMSE full: {rmse_full:.4f}, Naïve RMSE: {naive_full:.4f}")

    rmse_latent, naive_latent = lstm_results_latent(feature_idx=0)
    print(f"RMSE latent: {rmse_latent:.4f}, Naïve RMSE: {naive_latent:.4f}")