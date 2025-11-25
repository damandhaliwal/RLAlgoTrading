import pandas as pd
import numpy as np
import os
import pickle

from ivs_create import create_ivs
from utils import paths
from autoencoder import Autoencoder

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class IVS_Predictor(nn.Module):
    """LSTM predictor for IVS sequences."""

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
    """Load pretrained autoencoder and scaler."""
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
    """Create input/output sequences for LSTM training."""
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
    """
    Train LSTM predictor for IVS forecasting.

    Args:
        autoencoder: If True, predict in latent space (24-dim).
                     If False, predict full surface (374-dim).
        load_autoencoder: If True, load pretrained autoencoder.
        n_epochs: If 0, just load existing model without training.

    Returns:
        predictor: Trained LSTM model
        scaler: Scaler used for data normalization
        ae_model: Autoencoder model (or None if autoencoder=False)
    """
    path_dict = paths()
    models_dir = path_dict['models']
    os.makedirs(models_dir, exist_ok=True)

    # Keep original naming convention
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

    # === If n_epochs=0, just load existing model ===
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

    # === Create sequences ===
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

    # === Initialize model ===
    predictor = IVS_Predictor(input_dim, hidden_dim, num_layers)
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # === Training loop ===
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
    """Train both latent and full surface predictors."""

    print("\n" + "=" * 60)
    print("Training LATENT predictor (32-dim)")
    print("=" * 60)

    train_predictor(n_epochs=n_epochs, autoencoder=True, load_autoencoder=True)

    print("\n" + "=" * 60)
    print("Training FULL SURFACE predictor (374-dim)")
    print("=" * 60)

    train_predictor(n_epochs=n_epochs, autoencoder=False)


if __name__ == "__main__":
    train_both_predictors(n_epochs=200)