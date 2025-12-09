# Autoencoder definition
# Daman Dhaliwal

# import libraries
import os
from ivs_create import create_ivs
from utils import paths, load_data
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# define the autoencoder class - useful for accessing the hidden layers that will be useful for RL later
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
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


def train_autoencoder(overwrite = False, epochs = 100):
    # load ivs
    data, _ = create_ivs(overwrite = overwrite)
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

    # training loop
    loss = nn.MSELoss()
    model = Autoencoder(input_dim=input_dim, latent_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = epochs

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

    # save the model
    path = paths()
    models_dir = path['models']
    os.makedirs(models_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(models_dir, 'autoencoder.pth'))
    with open(os.path.join(models_dir, 'autoencoder_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {models_dir}")

    return model, scaler


# present the results
def autoencoder_results(sample_idx=10):
    path = paths()

    # load data
    data, dates = create_ivs(overwrite=False)
    data_flat = data.reshape((data.shape[0], -1))
    split_index = int(0.8 * data.shape[0])
    test_data = data_flat[split_index:]

    input_dim = data_flat.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=32)
    model.load_state_dict(torch.load(os.path.join(path['models'], 'autoencoder.pth')))
    model.eval()

    with open(os.path.join(path['models'], 'autoencoder_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    test_tensor = torch.tensor(scaler.transform(test_data), dtype=torch.float32)
    criterion = nn.MSELoss()

    with torch.no_grad():
        encoded, decoded = model(test_tensor)
        mse = criterion(decoded, test_tensor).item()

    target_idx = split_index + sample_idx
    original_raw = data[target_idx]

    input_vec = torch.tensor(scaler.transform(original_raw.reshape(1, -1)), dtype=torch.float32)
    with torch.no_grad():
        _, decoded_vec = model(input_vec)

    reconstructed = scaler.inverse_transform(decoded_vec.numpy()).reshape(original_raw.shape)

    # We load the raw dataframe to get the actual Days and Delta values
    df = load_data()
    # Apply the same filters as ivs_create.py to ensure alignment
    df = df[(((df['delta'] < 0) & (df['cp_flag'] == "P")) |
             ((df['delta'] > 0) & (df['cp_flag'] == "C")))]

    completeness = df.groupby('date').size()
    valid_dates = completeness[completeness == 374].index
    df = df[df['date'].isin(valid_dates)]

    unique_deltas = sorted(df['delta'].unique())
    unique_days = sorted(df['days'].unique())

    X, Y = np.meshgrid(unique_days, unique_deltas)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, original_raw, cmap=cm.viridis, edgecolor='none')
    ax1.set_title(f'Original ({dates[target_idx]})')
    ax1.set_xlabel('Maturity (Days)')
    ax1.set_ylabel('Delta')
    ax1.set_zlabel('Implied Volatility')
    ax1.invert_xaxis()  # Optional: standard for surface plots to show near-term at front

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, reconstructed, cmap=cm.inferno, edgecolor='none')
    ax2.set_title(f'Reconstructed')
    ax2.set_xlabel('Maturity (Days)')
    ax2.set_ylabel('Delta')
    ax2.set_zlabel('Implied Volatility')
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(path['plots'], 'figure4.png'), dpi=600)
    plt.close()

    return mse, np.sqrt(mse)

if __name__ == "__main__":
    mse, rmse = autoencoder_results(sample_idx=10)
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")