import numpy as np


# Initializes our LSTM network.
def init_lstm(hidden_size, vocab_size):
    input_size = vocab_size
    hidden_size = hidden_size

    # Forget gate weight and bias
    W_f = np.random.randn(hidden_size, input_size + hidden_size)
    b_f = np.zeros((hidden_size, 1))

    # Input gate weight and bias
    W_i = np.random.randn(hidden_size, input_size + hidden_size)
    b_i = np.zeros((hidden_size, 1))

    # Cell gate weight and bias
    W_c = np.random.randn(hidden_size, input_size + hidden_size)
    b_c = np.zeros((hidden_size, 1))

    # Output gate weight and bias
    W_o = np.random.randn(hidden_size, input_size + hidden_size)
    b_o = np.zeros((hidden_size, 1))

    return W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o
