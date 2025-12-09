import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from ivs_create import create_ivs
from utils import load_data, paths

def plot_spy_ivs():
    ivs_data, ivs_dates = create_ivs(overwrite=False)

    df = load_data()
    df = df[
        (((df['delta'] < 0) & (df['cp_flag'] == "P")) |
         ((df['delta'] > 0) & (df['cp_flag'] == "C")))
    ]

    unique_deltas = sorted(df['delta'].unique())
    unique_days = sorted(df['days'].unique())

    date_index = -1
    surface_slice = ivs_data[date_index]
    selected_date = pd.to_datetime(ivs_dates[date_index]).strftime('%Y-%m-%d')

    X, Y = np.meshgrid(unique_days, unique_deltas)
    Z = surface_slice

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.8)

    ax.set_xlabel('Days to Maturity')
    ax.set_ylabel('Delta')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Implied Volatility Surface - SPY')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=30, azim=-45)

    path_dict = paths()
    save_path = os.path.join(path_dict['plots'], 'figure2.png')
    os.makedirs(path_dict['plots'], exist_ok=True)

    plt.savefig(save_path, dpi=600)
    plt.close()

    return

if __name__ == "__main__":
    plot_spy_ivs()