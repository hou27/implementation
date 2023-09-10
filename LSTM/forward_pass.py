# def forward_pass(inputs, h_prev, C_prev, p):
#     """
#     Arguments:
#     x -- your input data at timestep "t", numpy array of shape (n_x, m).
#     h_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
#     C_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
#     p -- python list containing:
#                         W_f -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
#                         b_f -- Bias of the forget gate, numpy array of shape (n_a, 1)
#                         W_i -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
#                         b_i -- Bias of the update gate, numpy array of shape (n_a, 1)
#                         W_g -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
#                         b_g --  Bias of the first "tanh", numpy array of shape (n_a, 1)
#                         W_o -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
#                         b_o --  Bias of the output gate, numpy array of shape (n_a, 1)
#                         W_y -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
#                         b_y -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
#     Returns:
#     z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s -- lists of size m containing the computations in each forward pass
#     outputs -- prediction at timestep "t", numpy array of shape (n_y, m)
#     """
#     assert h_prev.shape == (hidden_size, 1)
#     assert C_prev.shape == (hidden_size, 1)

#     # First we unpack our parameters
#     W_f, W_i, W_g, W_o, W_y, b_f, b_i, b_g, b_o, b_y = p

#     # Save a list of computations for each of the components in the LSTM
#     (
#         x_s,
#         z_s,
#         f_s,
#         i_s,
#     ) = (
#         [],
#         [],
#         [],
#         [],
#     )
#     g_s, C_s, o_s, h_s = [], [], [], []
#     v_s, output_s = [], []

#     # Append the initial cell and hidden state to their respective lists
#     h_s.append(h_prev)
#     C_s.append(C_prev)

#     for x in inputs:
#         # YOUR CODE HERE!
#         # Concatenate input and hidden state
#         z = np.row_stack((h_prev, x))
#         z_s.append(z)

#         # YOUR CODE HERE!
#         # Calculate forget gate
#         f = sigmoid(np.dot(W_f, z) + b_f)
#         f_s.append(f)

#         # Calculate input gate
#         i = sigmoid(np.dot(W_i, z) + b_i)
#         i_s.append(i)

#         # Calculate candidate
#         g = tanh(np.dot(W_g, z) + b_g)
#         g_s.append(g)

#         # YOUR CODE HERE!
#         # Calculate memory state
#         C_prev = f * C_prev + i * g
#         C_s.append(C_prev)

#         # Calculate output gate
#         o = sigmoid(np.dot(W_o, z) + b_o)
#         o_s.append(o)

#         # Calculate hidden state
#         h_prev = o * tanh(C_prev)
#         h_s.append(h_prev)

#         # Calculate logits
#         v = np.dot(W_y, h_prev) + b_y
#         v_s.append(v)

#         # Calculate softmax
#         output = softmax(v)
#         output_s.append(output)

#     return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s


# # Get first sentence in test set
# inputs, targets = test_set[1]

# # One-hot encode input and target sequence
# inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
# targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

# # Initialize hidden state as zeros
# h = np.zeros((hidden_size, 1))
# c = np.zeros((hidden_size, 1))

# # Forward pass
# z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params)

# output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
# print("Input sentence:")
# print(inputs)

# print("\nTarget sequence:")
# print(targets)

# print("\nPredicted sequence:")
# print([idx_to_word[np.argmax(output)] for output in outputs])

import sys

sys.path.append("/Users/hou27/workspace/ml/with_numpy")

import numpy as np
from activation.sigmoid import sigmoid
from activation.tanh import tanh


def forward_pass(x, h_prev, c_prev, params):
    W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o = params

    # Concatenate input and hidden state
    x_concat = np.concatenate((h_prev, x), axis=1)

    # Update forget gate
    f_t = sigmoid(np.dot(W_f, x_concat.T) + b_f)

    # Update input gate
    i_t = sigmoid(np.dot(W_i, x_concat.T) + b_i)

    # Update cell gate
    c_candidate_t = tanh(np.dot(W_c, x_concat.T) + b_c)
    c_t = f_t * c_prev + i_t * c_candidate_t

    # Update output gate
    o_t = sigmoid(np.dot(W_o, x_concat.T) + b_o)

    # Update hidden state
    h_t = o_t * tanh(c_t)

    return h_t, c_t
