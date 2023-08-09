import numpy as np

# Initialize recurrent neural network
def init_rnn(hidden_size, vocab_size):
    """
    정규 분포를 따르는 무작의 값으로 파라미터 초기화
    """
    # Weight matrix (input to hidden state)
    W_xh = np.random.randn((hidden_size, vocab_size))

    # Weight matrix (recurrent computation)
    W_hh = np.random.randn((hidden_size, hidden_size))

    # Weight matrix (hidden state to output)
    W_hy = np.random.randn((vocab_size, hidden_size))

    # Bias (hidden state)
    b_hidden = np.zeros((hidden_size, 1))

    # Bias (output)
    b_out = np.zeros((vocab_size, 1))

    return W_xh, W_hh, W_hy, b_hidden, b_out