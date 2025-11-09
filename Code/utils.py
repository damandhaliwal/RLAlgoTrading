import os
import pandas as pd

# get clean paths
def paths():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(code_dir)

    data_dir = os.path.join(parent_dir, 'Data/')
    data_output_dir = os.path.join(parent_dir, 'Output', 'Data/')
    plots_dir = os.path.join(parent_dir, 'Output', 'Plots/')
    tables_dir = os.path.join(parent_dir, 'Output', 'Tables/')

    return {
        'parent_dir': parent_dir,
        'data_input': data_dir,
        'data': data_output_dir,
        'plots': plots_dir,
        'tables': tables_dir
    }

# load and clean data
def load_data():
    path = paths()
    data = pd.read_csv(path['data_input'] + 'data.csv')

    # clean NaNs and zeros
    data = data.drop(['issue_type', 'class', 'industry_group', 'sic'], axis=1)
    data = data.dropna()

    # keep SPY only
    data = data[data['ticker'] == 'SPY']

    # save the data as csv
    data.to_csv(path['data'] + 'data_clean_SPY.csv', index=False)

    return data