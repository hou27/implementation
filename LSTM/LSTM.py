import numpy as np


class LSTM:
    def __init__(self, hidden_size, vocab_size, bptt_truncate=4):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        (
            self.W_f,
            self.W_i,
            self.W_g,
            self.W_o,
            self.W_v,
            self.b_f,
            self.b_i,
            self.b_g,
            self.b_o,
            self.b_v,
        ) = self.init_lstm(hidden_size, vocab_size)

    def init_lstm(hidden_size, vocab_size, z_size):
        """
        Initializes our LSTM network.

        Args:
        `hidden_size`: the dimensions of the hidden state
        `vocab_size`: the dimensions of our vocabulary
        """
        # Weight matrix (forget gate)
        W_f = np.random.randn(hidden_size, z_size)

        # Bias for forget gate
        b_f = np.zeros((hidden_size, 1))

        # Weight matrix (input gate)
        W_i = np.random.randn(hidden_size, z_size)

        # Bias for input gate
        b_i = np.zeros((hidden_size, 1))

        # Weight matrix (candidate)
        W_g = np.random.randn(hidden_size, z_size)

        # Bias for candidate
        b_g = np.zeros((hidden_size, 1))

        # Weight matrix of the output gate
        W_o = np.random.randn(hidden_size, z_size)
        b_o = np.zeros((hidden_size, 1))

        # Weight matrix relating the hidden-state to the output
        W_v = np.random.randn(vocab_size, hidden_size)
        b_v = np.zeros((vocab_size, 1))

        return W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v

    def forward_pass(inputs, h_prev, C_prev, p):
        """
        Arguments:
        x -- your input data at timestep "t", numpy array of shape (n_x, m).
        h_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        C_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
        p -- python list containing:
                            W_f -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            b_f -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            W_i -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            b_i -- Bias of the update gate, numpy array of shape (n_a, 1)
                            W_g -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            b_g --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                            W_o -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            b_o --  Bias of the output gate, numpy array of shape (n_a, 1)
                            W_v -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_v, n_a)
                            b_v -- Bias relating the hidden-state to the output, numpy array of shape (n_v, 1)
        Returns:
        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s -- lists of size m containing the computations in each forward pass
        outputs -- prediction at timestep "t", numpy array of shape (n_v, m)
        """
        assert h_prev.shape == (hidden_size, 1)
        assert C_prev.shape == (hidden_size, 1)

        # First we unpack our parameters
        W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p

        # Save a list of computations for each of the components in the LSTM
        (
            x_s,
            z_s,
            f_s,
            i_s,
        ) = (
            [],
            [],
            [],
            [],
        )
        g_s, C_s, o_s, h_s = [], [], [], []
        v_s, output_s = [], []

        # Append the initial cell and hidden state to their respective lists
        h_s.append(h_prev)
        C_s.append(C_prev)

        for x in inputs:
            # YOUR CODE HERE!
            # Concatenate input and hidden state
            z = np.row_stack((h_prev, x))
            z_s.append(z)

            # YOUR CODE HERE!
            # Calculate forget gate
            f = sigmoid(np.dot(W_f, z) + b_f)
            f_s.append(f)

            # Calculate input gate
            i = sigmoid(np.dot(W_i, z) + b_i)
            i_s.append(i)

            # Calculate candidate
            g = tanh(np.dot(W_g, z) + b_g)
            g_s.append(g)

            # YOUR CODE HERE!
            # Calculate memory state
            C_prev = f * C_prev + i * g
            C_s.append(C_prev)

            # Calculate output gate
            o = sigmoid(np.dot(W_o, z) + b_o)
            o_s.append(o)

            # Calculate hidden state
            h_prev = o * tanh(C_prev)
            h_s.append(h_prev)

            # Calculate logits
            v = np.dot(W_v, h_prev) + b_v
            v_s.append(v)

            # Calculate softmax
            output = softmax(v)
            output_s.append(output)

        return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s
