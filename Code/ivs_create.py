# import libraries
import pandas as pd
import numpy as np
import os
from utils import paths, load_data


def create_ivs(overwrite=False):
    path = paths()
    ivs_filepath = os.path.join(path['data'], 'ivs.npy')
    dates_filepath = os.path.join(path['data'], 'ivs_dates.npy')

    if not overwrite and os.path.exists(ivs_filepath) and os.path.exists(dates_filepath):
        print("Loading IVS and Dates from cache...")
        ivs = np.load(ivs_filepath)
        dates = np.load(dates_filepath, allow_pickle=True)
        return ivs, pd.to_datetime(dates)

    df = load_data()

    df = df[
        (((df['delta'] < 0) & (df['cp_flag'] == "P")) | ((df['delta'] > 0) & (df['cp_flag'] == "C")))
    ]

    completeness = df.groupby('date').size()
    valid_dates = completeness[completeness == 374].index
    df = df[df['date'].isin(valid_dates)]

    # Pivot
    ivs_pivot = df.pivot_table(values='impl_volatility', index=['date', 'delta'], columns=['days'])

    unique_dates = ivs_pivot.index.get_level_values('date').unique()
    unique_deltas = ivs_pivot.index.get_level_values('delta').unique()
    unique_days = ivs_pivot.columns.get_level_values('days').unique()

    # Convert to Numpy
    ivs = ivs_pivot.to_numpy()
    ivs = ivs.reshape((len(unique_dates), len(unique_deltas), len(unique_days)))

    # Ensure dates are sorted and formatted
    dates = pd.to_datetime(unique_dates).sort_values()

    # 3. Save to Cache
    np.save(ivs_filepath, ivs)
    np.save(dates_filepath, dates.to_numpy())

    return ivs, dates