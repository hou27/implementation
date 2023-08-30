import matplotlib.pyplot as plt
import numpy as np

from backward_pass import backward_pass
from forward_pass import forward_pass
from init import init_rnn
from update_params import update_params

hidden_size = 1
input_dims = 1

if __name__ == "__main__":
    X = np.array(
        [
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            [[2.0], [3.0], [4.0], [5.0], [6.0]],
            [[3.0], [4.0], [5.0], [6.0], [7.0]],
            [[4.0], [5.0], [6.0], [7.0], [8.0]],
            [[5.0], [6.0], [7.0], [8.0], [9.0]],
        ]
    )
    y = np.array([[6.0], [7.0], [8.0], [9.0], [10.0]])

    num_epochs = 1000
    learning_rate = 3e-4

    # 네트워크 초기화
    params = init_rnn(hidden_size=hidden_size, input_dims=input_dims)

    # hidden state 0으로 초기화
    hidden_state = np.zeros((hidden_size, 1))

    # Track loss
    training_loss = []

    # For each epoch
    for i in range(num_epochs):
        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0

        # inputs = X
        # targets = y
        inputs = np.array([1, 2, 3, 4])
        targets = np.array([2, 3, 4, 5])

        # 매 epoch마다 hidden state를 0으로 초기화
        hidden_state = np.zeros_like(hidden_state)

        # Forward pass
        outputs, hidden_states = forward_pass(inputs, hidden_state, params)

        # Backward pass
        loss, grads = backward_pass(inputs, outputs, hidden_states, targets, params)

        # Update parameters
        params = update_params(params, grads, learning_rate=learning_rate)

        # Update loss
        epoch_training_loss += loss

        # plot으로
        training_loss.append(epoch_training_loss / len(X))

        # Loss 출력
        print(f"Epoch {i}, training loss: {training_loss[-1]}")

    inputs = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    )
    targets = np.array(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    )

    # hidden_state = np.zeros((hidden_size, 1))
    hidden_state = np.zeros((1))

    # Forward pass
    outputs, hidden_states = forward_pass(inputs, hidden_state, params)

    print("Input :")
    print(inputs)

    print("\nTarget :")
    print(targets)

    print("\nPredicted :")
    print(outputs)

    # Plot으로 loss 확인
    epoch = np.arange(len(training_loss))
    plt.figure()
    plt.plot(
        epoch,
        training_loss,
        "r",
        label="Training loss",
    )
    plt.legend()
    plt.xlabel("Epoch"), plt.ylabel("RMSE")  # RMSE
    plt.show()
