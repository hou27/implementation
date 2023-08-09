import sys

sys.path.append("/Users/hou27/workspace/ml/with_numpy")

import numpy as np
from activation.softmax import softmax
from activation.tanh import tanh

# Vanilla RNN forward pass
def forward_pass(inputs, hidden_state, params):
    # 파라미터 추출
    W_xh, W_hh, W_hy, b_hidden, b_out = params # 입력 가중치 행렬, hidden 가중치 행렬, 출력 가중치 행렬, hidden bias, 출력 bias
    
    # outputs, hidden_states 초기화
    outputs, hidden_states = [], []
    
    # For each element in input sequence
    for t in range(len(inputs)):

        # hidden_state 계산
        hidden_state = tanh(np.dot(W_hh, hidden_state) + np.dot(W_xh, inputs[t]) + b_hidden)

        # output 계산
        out = softmax(np.dot(W_hy, hidden_state) + b_out)
        
        # 저장
        outputs.append(out)
        hidden_states.append(hidden_state.copy())
    
    return outputs, hidden_states