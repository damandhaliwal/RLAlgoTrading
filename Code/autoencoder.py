# import libraries
import pandas as pd
import numpy as np
import os
from ivs_create import create_ivs
from utils import paths
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_autoencoder():
    # load ivs
    data = create_ivs(overwrite=False)
    data = data.reshape((data.shape[0], -1))

    # split into train and test dataset
    split_index = int(0.8 * data.shape[0])
    train_data = data[:split_index]
    test_data = data[split_index:]

    # standardize the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor, train_tensor)
    test_dataset = TensorDataset(test_tensor, test_tensor)

    batch_size = 64

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get the number of features for the input layer
    input_dim = train_tensor.shape[1]

    # define the autoencoder class - useful for accessing the hidden layers that will be useful for RL later
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=16):
            super(Autoencoder, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, latent_dim)
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    # training loop
    loss = nn.MSELoss()
    model = Autoencoder(input_dim=input_dim, latent_dim=24)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0

        for batch_data, _ in train_loader:
            encoded, decoded = model(batch_data)
            loss_value = loss(decoded, batch_data)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            total_train_loss += loss_value.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Train Loss: {avg_train_loss}")

    # test loop
    model.eval()
    total_test_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch_data, _ in test_loader:
            encoded, decoded = model(batch_data)
            loss_value = loss(decoded, batch_data)
            total_test_loss += loss_value.item()
            num_batches += 1
    avg_test_loss = total_test_loss / num_batches
    print(f"Average Test Loss: {avg_test_loss}")

    return model, scaler

