# import libraries
import pandas as pd
import numpy as np
import os
from utils import paths, load_data

path = paths()

# main body to create ivs - will convert to function later
def create_ivs(overwrite = False):
    filepath = path['data'] + 'ivs.npy'

    if not overwrite and os.path.exists(filepath):
        ivs = np.load(filepath)
        return ivs

    df = load_data()
    df = df[
        (((df['delta'] < 0) & (df['cp_flag'] == "P")) | ((df['delta'] > 0) & (df['cp_flag'] == "C")))
    ]
    completeness = df.groupby('date').size()
    a = completeness[completeness == 374]
    df = df[df['date'].isin(a.index)]
    ivs = df.pivot_table(values = 'impl_volatility', index = ['date', 'delta'], columns = ['days'])

    unique_dates = ivs.index.get_level_values('date').unique()
    unique_deltas = ivs.index.get_level_values('delta').unique()
    unique_days = ivs.columns.get_level_values('days').unique()

    ivs = ivs.to_numpy()
    ivs = ivs.reshape((len(unique_dates), len(unique_deltas), len(unique_days)))

    np.save(filepath, ivs)

    return ivs

