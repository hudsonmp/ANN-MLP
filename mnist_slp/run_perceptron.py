import numpy as np
from single_layer_perceptron.mnist_exploration import load_data
from single_layer_perceptron.perceptron import Perceptron
from single_layer_perceptron.single_layer import (
    SingleLayerForward,
    SingleLayerBackward,
    train_step,
)


def main():
    # Load MNIST data
    print("Loading MNIST data...")
    X, y = load_data()

    # Preprocess data
    print("Preprocessing data...")
    X = X / 255.0  # Normalize pixel values to [0, 1]

    # Convert labels to binary (0 vs non-0)
    y_binary = (y == "0").astype(int)

    # Split data into train and test sets
    train_size = 50000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_binary[:train_size], y_binary[train_size:]

    # Initialize single layer network
    input_size = X.shape[1]  # 784 features
    output_size = 1  # Binary classification
    forward_layer = SingleLayerForward(input_size, output_size)
    backward_layer = SingleLayerBackward()

    # Training parameters
    epochs = 10
    batch_size = 32
    learning_rate = 0.01

    print(f"Network architecture:")
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")
    print(f"Weight shape: {forward_layer.weights.shape}")
    print(f"Bias shape: {forward_layer.biases.shape}")

    print("\nStarting training...")
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Train in batches
        num_batches = len(X_train) // batch_size
        total_loss = 0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # Prepare batch data
            # X_batch shape: (784, batch_size)
            X_batch = X_shuffled[start_idx:end_idx].T
            # y_batch shape: (1, batch_size)
            y_batch = y_shuffled[start_idx:end_idx].reshape(1, -1)

            if i == 0 and epoch == 0:
                print(f"\nFirst batch shapes:")
                print(f"X_batch shape: {X_batch.shape}")
                print(f"y_batch shape: {y_batch.shape}")
                print(f"Expected shapes in forward pass:")
                print(f"  Input X: {X_batch.shape}")
                print(f"  Weights: {forward_layer.weights.shape}")
                print(f"  Biases: {forward_layer.biases.shape}")

            # Perform one training step
            train_step(X_batch, y_batch, forward_layer, backward_layer, learning_rate)

            # Calculate batch loss (optional)
            predictions, _ = forward_layer.forward(X_batch)
            batch_loss = -np.mean(
                y_batch * np.log(predictions + 1e-10)
                + (1 - y_batch) * np.log(1 - predictions + 1e-10)
            )
            total_loss += batch_loss

        # Calculate average loss and accuracy on training set
        avg_loss = total_loss / num_batches
        _, train_predictions = forward_layer.forward(X_train.T)
        train_accuracy = np.mean(
            (train_predictions > 0.5).astype(int).flatten() == y_train
        )

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}"
        )

    # Evaluate on test set
    _, test_predictions = forward_layer.forward(X_test.T)
    test_accuracy = np.mean((test_predictions > 0.5).astype(int).flatten() == y_test)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
