# main script to reproduce figures and tables used in the paper
# import functions
from autoencoder import autoencoder_results
from plots import plot_spy_ivs
from LSTM import lstm_results_full

# figure 2
plot_spy_ivs()

# figure 4
autoencoder_results(sample_idx = 10) # change sample_idx to reproduce different days

# figure 5
lstm_results_full()

