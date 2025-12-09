# main script to reproduce figures and tables used in the paper
# import functions
from synthetic_pipeline import run_synthetic_pipeline
from autoencoder import autoencoder_results
from plots import plot_spy_ivs
from LSTM import lstm_results_full
from utils import paths
from RL import generate_final_table
from robustness import run_synthetic_robustness

# figure 2
plot_spy_ivs()

# figure 4
autoencoder_results(sample_idx = 10) # change sample_idx to reproduce different days

# figure 5
lstm_results_full()

# table 1
generate_final_table(n_epochs=100)

# table 2
path_dict = paths()
model_name = 'agent_RNNFNN_hybrid_pred_put.pth'
full_path = os.path.join(path_dict['models'], model_name)

run_synthetic_robustness(full_path)

# table 3
run_synthetic_pipeline()