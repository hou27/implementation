import numpy as np
import matplotlib.pyplot as plt
from backward_pass import backward_pass
from forward_pass import forward_pass
from data import (
    Dataset,
    create_datasets,
    generate_dataset,
    one_hot_encode_sequence,
    sequences_to_dicts,
)
from init import init_lstm
from update_params import update_params

sequences = generate_dataset()
word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences)
training_set, validation_set, test_set = create_datasets(sequences, Dataset)

# Hyper-parameters
num_epochs = 50
hidden_size = 50

# Initialize a new network
params = init_lstm(hidden_size=hidden_size, vocab_size=vocab_size)

# Initialize hidden state as zeros
hidden_state = np.zeros((hidden_size, 1))

# Track loss
training_loss, validation_loss = [], []

# For each epoch
for i in range(num_epochs):
    # Track loss
    epoch_training_loss = 0
    epoch_validation_loss = 0

    # For each sentence in validation set
    for inputs, targets in validation_set:
        # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

        # Initialize hidden state and cell state as zeros
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        # Forward pass
        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward_pass(
            inputs_one_hot, h, c, params
        )

        # Backward pass
        loss, _ = backward_pass(
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params
        )

        # Update loss
        epoch_validation_loss += loss

    # For each sentence in training set
    for inputs, targets in training_set:
        # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

        # Initialize hidden state and cell state as zeros
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        # Forward pass
        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward_pass(
            inputs_one_hot, h, c, params
        )

        # Backward pass
        loss, grads = backward_pass(
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params
        )

        # Update parameters
        params = update_params(params, grads, lr=1e-1)

        # Update loss
        epoch_training_loss += loss

    # Save loss for plot
    training_loss.append(epoch_training_loss / len(training_set))
    validation_loss.append(epoch_validation_loss / len(validation_set))

    # Print loss every 5 epochs
    if i % 5 == 0:
        print(
            f"Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}"
        )


# Get first sentence in test set
inputs, targets = test_set[1]

# One-hot encode input and target sequence
inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

# Initialize hidden state as zeros
h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))

# Forward pass
z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward_pass(
    inputs_one_hot, h, c, params
)

# Print example
print("Input sentence:")
print(inputs)

print("\nTarget sequence:")
print(targets)

print("\nPredicted sequence:")
print([idx_to_word[np.argmax(output)] for output in outputs])

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(
    epoch,
    training_loss,
    "r",
    label="Training loss",
)
plt.plot(epoch, validation_loss, "b", label="Validation loss")
plt.legend()
plt.xlabel("Epoch"), plt.ylabel("NLL")
plt.show()
