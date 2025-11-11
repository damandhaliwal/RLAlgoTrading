# import libaries
import pandas as pd
import numpy as np
import os

from ivs_create import create_ivs
from utils import paths
from autoencoder import train_autoencoder

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_predictor(sequence_length = 5, hidden_dim = 32, num_layers = 2, n_epochs = 200, overwrite = False, autoencoder = True):
    scaler = None
    if autoencoder is True:
        model_ae, scaler = train_autoencoder(overwrite = overwrite)

        data = create_ivs(overwrite = overwrite)
        data = data.reshape((data.shape[0], -1))
        data = scaler.transform(data)

        # extract latent representations
        model_ae.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype = torch.float32)
            latent_data = model_ae.encoder(data_tensor)
            latent_data = latent_data.numpy()
    else:
        data = create_ivs(overwrite = overwrite)
        data = data.reshape((data.shape[0], -1))

        # Scale the data even without autoencoder
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        latent_data = data

    # function to create sequences for LSTM input
    def create_sequences(data, sequence_length):
        num_sequences = len(data) - sequence_length
        latent_dim = data.shape[1]

        # pre-allocate arrays
        X = np.zeros((num_sequences, sequence_length, latent_dim))
        y = np.zeros((num_sequences, latent_dim))

        for i in range(num_sequences):
            X[i] = data[i:i + sequence_length]
            y[i] = data[i + sequence_length]

        return X, y

    X, y = create_sequences(latent_data, sequence_length = sequence_length)

    # split into train and test dataset
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # IVS predictor
    class IVS_predictor(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super(IVS_predictor, self).__init__()

            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                                dropout=0.3)

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

    # training loop
    loss = nn.MSELoss()
    model = IVS_predictor(input_dim=X.shape[2], hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        num_batches_train = 0

        for batch_X, batch_y in train_loader:
            predicted = model(batch_X)
            loss_value = loss(predicted, batch_y)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            total_train_loss += loss_value.item()
            num_batches_train += 1

        avg_train_loss = total_train_loss / num_batches_train

        # validation loop
        model.eval()
        total_test_loss = 0
        num_batches_test = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predicted = model(batch_X)
                loss_value = loss(predicted, batch_y)

                total_test_loss += loss_value.item()
                num_batches_test += 1

            avg_test_loss = total_test_loss / num_batches_test

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

    return model, scaler