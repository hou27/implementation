import numpy as np

# Initialize recurrent neural network
def init_rnn(hidden_size, input_dims):
    """
    정규 분포를 따르는 무작위 값으로 파라미터 초기화
    """
    # Weight matrix (input to hidden state)
    W_xh = np.random.randn(hidden_size, input_dims)

    # Weight matrix (recurrent computation)
    W_hh = np.random.randn(hidden_size, hidden_size)

    # Weight matrix (hidden state to output)
    W_hy = np.random.randn(input_dims, hidden_size)

    # Bias (hidden state)
    b_hidden = np.zeros((hidden_size, 1))

    # Bias (output)
    b_out = np.zeros((input_dims, 1))

    return W_xh, W_hh, W_hy, b_hidden, b_out