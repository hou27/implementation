import sys

sys.path.append("/Users/hou27/workspace/ml/with_numpy")

import numpy as np
from activation.tanh import d_tanh


# Backward pass for a vanilla RNN cell.
def backward_pass(inputs, outputs, hidden_states, targets, params):
    W_xh, W_hh, W_hy, b_hidden, b_out = params

    # Initialize gradients 0으로 초기화
    d_xh, d_hh, d_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
    d_b_hidden, d_b_out = np.zeros_like(b_hidden), 0

    # 다음 hidden state에 대한 gradient
    d_h_next = 0

    # Loss
    loss = 0

    # output을 역순으로 순회
    # NB: We iterate backwards s.t. t = N, N-1, ... 1, 0
    for t in reversed(range(len(outputs))):
        # Compute RMSE
        loss += np.sqrt(np.mean(np.square(outputs[t] - targets[t])))

        # Backpropagate into output
        d_y_t = outputs[t] - targets[t]  # = 손실함수 RMSE를 출력 오차(y_pred - y_true)로 미분한 값

        # 현재 타임스텝에서의 출력 오차(d_y_t)와 hidden state 값을 곱하여 W_hy에 대한 gradient의 계산
        d_hy += (
            d_y_t * hidden_states[t]
        )  # (출력 y = W_hy * h + b를 W_hy에 대해 미분한 값) X (d_y_t)
        d_b_out += d_y_t

        # Backpropagate into h_t
        # hidden node의 gradient는 이전 hidden node의 gradient와 현재 output의 gradient의 합으로 계산
        d_h_t = np.dot(W_hy.T, d_y_t) + d_h_next

        # Backpropagate (비선형인 점 주의)
        # 흘러들어온 gradient에 대해 tanh의 미분값을 곱해줌
        d_h_raw = d_tanh(hidden_states[t]) * d_h_t
        d_b_hidden += d_h_raw

        # Backpropagate into W_xh
        d_xh += np.dot(d_h_raw, inputs[t].T)

        # Backpropagate into W_hh
        d_hh += np.dot(d_h_raw, hidden_states[t - 1].T)
        d_h_next = np.dot(W_hh.T, d_h_raw)  # 다음 hidden node의 gradient를 계산하기 위해 저장

    # Pack gradients
    grads = d_xh, d_hh, d_hy, d_b_hidden, d_b_out

    return loss, grads
