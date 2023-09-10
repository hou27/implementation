# def backward(z, f, i, g, C, o, h, v, outputs, targets, p=params):
#     """
#     Arguments:
#     z -- your concatenated input data  as a list of size m.
#     f -- your forget gate computations as a list of size m.
#     i -- your input gate computations as a list of size m.
#     g -- your candidate computations as a list of size m.
#     C -- your Cell states as a list of size m+1.
#     o -- your output gate computations as a list of size m.
#     h -- your Hidden state computations as a list of size m+1.
#     v -- your logit computations as a list of size m.
#     outputs -- your outputs as a list of size m.
#     targets -- your targets as a list of size m.
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
#     loss -- crossentropy loss for all elements in output
#     grads -- lists of gradients of every element in p
#     """

#     # Unpack parameters
#     W_f, W_i, W_g, W_o, W_y, b_f, b_i, b_g, b_o, b_y = p

#     # Initialize gradients as zero
#     W_f_d = np.zeros_like(W_f)
#     b_f_d = np.zeros_like(b_f)

#     W_i_d = np.zeros_like(W_i)
#     b_i_d = np.zeros_like(b_i)

#     W_g_d = np.zeros_like(W_g)
#     b_g_d = np.zeros_like(b_g)

#     W_o_d = np.zeros_like(W_o)
#     b_o_d = np.zeros_like(b_o)

#     W_y_d = np.zeros_like(W_y)
#     b_y_d = np.zeros_like(b_y)

#     # Set the next cell and hidden state equal to zero
#     dh_next = np.zeros_like(h[0])
#     dC_next = np.zeros_like(C[0])

#     # Track loss
#     loss = 0

#     for t in reversed(range(len(outputs))):
#         # Compute the cross entropy
#         loss += -np.mean(np.log(outputs[t]) * targets[t])
#         # Get the previous hidden cell state
#         C_prev = C[t - 1]

#         # Compute the derivative of the relation of the hidden-state to the output gate
#         dv = np.copy(outputs[t])
#         dv[np.argmax(targets[t])] -= 1

#         # Update the gradient of the relation of the hidden-state to the output gate
#         W_y_d += np.dot(dv, h[t].T)
#         b_y_d += dv

#         # Compute the derivative of the hidden state and output gate
#         dh = np.dot(W_y.T, dv)
#         dh += dh_next
#         do = dh * tanh(C[t])
#         do = sigmoid(o[t], derivative=True) * do

#         # Update the gradients with respect to the output gate
#         W_o_d += np.dot(do, z[t].T)
#         b_o_d += do

#         # Compute the derivative of the cell state and candidate g
#         dC = np.copy(dC_next)
#         dC += dh * o[t] * tanh(tanh(C[t]), derivative=True)
#         dg = dC * i[t]
#         dg = tanh(g[t], derivative=True) * dg

#         # Update the gradients with respect to the candidate
#         W_g_d += np.dot(dg, z[t].T)
#         b_g_d += dg

#         # Compute the derivative of the input gate and update its gradients
#         di = dC * g[t]
#         di = sigmoid(i[t], True) * di
#         W_i_d += np.dot(di, z[t].T)
#         b_i_d += di

#         # Compute the derivative of the forget gate and update its gradients
#         df = dC * C_prev
#         df = sigmoid(f[t]) * df
#         W_f_d += np.dot(df, z[t].T)
#         b_f_d += df

#         # Compute the derivative of the input and update the gradients of the previous hidden and cell state
#         dz = (
#             np.dot(W_f.T, df)
#             + np.dot(W_i.T, di)
#             + np.dot(W_g.T, dg)
#             + np.dot(W_o.T, do)
#         )
#         dh_prev = dz[:hidden_size, :]
#         dC_prev = f[t] * dC

#     grads = W_f_d, W_i_d, W_g_d, W_o_d, W_y_d, b_f_d, b_i_d, b_g_d, b_o_d, b_y_d

#     # Clip gradients
#     grads = clip_gradient_norm(grads)

#     return loss, grads


# # Perform a backward pass
# loss, grads = backward(
#     z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params
# )

# print("We get a loss of:")
# print(loss)

import sys

sys.path.append("/Users/hou27/workspace/ml/with_numpy")

import numpy as np
from activation.tanh import tanh, d_tanh
from activation.sigmoid import sigmoid, d_sigmoid


def backward_pass(x, h_prev, c_prev, h_next, c_next, dh_next, dc_next, params):
    W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o = params

    # Concatenate input and hidden state
    x_concat = np.concatenate((h_prev, x), axis=1)

    # Compute activations
    i_t = sigmoid(np.dot(W_i, x_concat.T) + b_i)
    f_t = sigmoid(np.dot(W_f, x_concat.T) + b_f)
    o_t = sigmoid(np.dot(W_o, x_concat.T) + b_o)
    c_candidate_t = tanh(np.dot(W_c, x_concat.T) + b_c)

    # Compute gradients
    do_t = dh_next * tanh(c_next) * d_sigmoid(np.dot(W_o, x_concat.T) + b_o)
    dc_candidate_t = (dc_next + o_t * dh_next * d_tanh(c_next)) * i_t
    di_t = (
        (dc_next + o_t * dh_next * d_tanh(c_next))
        * c_candidate_t
        * d_sigmoid(np.dot(W_i, x_concat.T) + b_i)
    )
    df_t = (
        (dc_next + o_t * dh_next * d_tanh(c_next))
        * c_prev
        * d_sigmoid(np.dot(W_f, x_concat.T) + b_f)
    )

    # Compute weight and bias gradients
    dW_o = np.dot(x_concat.T, do_t.T)
    db_o = np.sum(do_t, axis=-1, keepdims=True)
    dW_c = np.dot(x_concat.T, dc_candidate_t.T)
    dbc = np.sum(dc_candidate_t, axis=-1, keepdims=True)
    dW_i = np.dot(x_concat.T, di_t.T)
    db_i = np.sum(di_t, axis=-1, keepdims=True)
    dW_f = np.dot(x_concat.T, df_t.T)
    db_f = np.sum(df_t, axis=-1, keepdims=True)

    # Compute input and hidden state gradients
    d_x_concat = (
        np.dot(W_o.T, do_t)
        + np.dot(W_c.T, dc_candidate_t)
        + np.dot(W_i.T, di_t)
        + np.dot(W_f.T, df_t)
    ).T
    dx = d_x_concat[:, hidden_size:]
    dh_prev = d_x_concat[:, :hidden_size]

    return dx, dh_prev, dc_next
