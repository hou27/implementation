import matplotlib.pyplot as plt
import numpy as np

from backward_pass import backward_pass
from forward_pass import forward_pass
from init import init_rnn
from update_params import update_params

hidden_size = 50
input_dims = 5

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

    num_epochs = 55
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

        inputs = X
        targets = y

        # 매 epoch마다 hidden state를 0으로 초기화
        hidden_state = np.zeros_like(hidden_state)

        # Forward pass
        outputs, hidden_states, _ = forward_pass(inputs, hidden_state, params)

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

    # Get first sentence in test set
    inputs = np.array(
        [
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            [[2.0], [3.0], [4.0], [5.0], [6.0]],
            [[3.0], [4.0], [5.0], [6.0], [7.0]],
            [[4.0], [5.0], [6.0], [7.0], [8.0]],
            [[5.0], [6.0], [7.0], [8.0], [9.0]],
            [[6.0], [7.0], [8.0], [9.0], [10.0]],
            [[7.0], [8.0], [9.0], [10.0], [11.0]],
            [[8.0], [9.0], [10.0], [11.0], [12.0]],
            [[9.0], [10.0], [11.0], [12.0], [13.0]],
            [[10.0], [11.0], [12.0], [13.0], [14.0]],
        ]
    )
    targets = np.array(
        [[6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0], [13.0], [14.0], [15.0]]
    )

    hidden_state = np.zeros((hidden_size, 1))

    # Forward pass
    outputs, hidden_states, actual_values = forward_pass(
        inputs, hidden_state, params, print_values=True
    )

    print("Input :")
    print(inputs)

    print("\nTarget :")
    print(targets)

    print("\nPredicted :")
    print(outputs)
    # output의 각 값의 합이 1이 되도록 softmax를 적용했기 때문에, argmax를 통해 가장 큰 값을 가져옴
    print([actual_values[i][np.argmax(outputs[i])] for i in range(len(outputs))])

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
    plt.xlabel("Epoch"), plt.ylabel("NLL")  # Negative Log Likelihood
    plt.show()
