import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from mnist_slp.single_layer_perceptron.single_layer import SingleLayerForward
from utils.mnist_loader import MNISTLoader


def batch_normalize(X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Apply batch normalization to input data."""
    mean = np.mean(X, axis=0, keepdims=True)
    var = np.var(X, axis=0, keepdims=True)
    return (X - mean) / np.sqrt(var + epsilon)


def main():
    """Quick training for initial model."""
    print("Loading MNIST data (subset for quick training)...")
    mnist_loader = MNISTLoader(subset_size=20000)  # Using 20k samples for quick training
    X_train, X_test, y_train, y_test = mnist_loader.load_mnist()

    print("Creating and training model...")
    model = SingleLayerForward(784, 10)
    
    # Training parameters
    n_epochs = 50  # Increased epochs
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9

    # Initialize momentum variables
    v_weights = np.zeros_like(model.weights)
    v_biases = np.zeros_like(model.biases)

    best_accuracy = 0
    best_weights = None
    best_biases = None

    # Training loop
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0
        n_batches = 0

        # Mini-batch training
        for i in range(0, len(X_train_shuffled), batch_size):
            batch_X = X_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]

            # Apply batch normalization
            batch_X = batch_normalize(batch_X)

            # Forward pass
            batch_X = batch_X.T
            batch_y = batch_y.T
            predictions, _ = model.forward(batch_X)

            # Compute loss
            loss = -np.mean(batch_y * np.log(predictions + 1e-8) + (1 - batch_y) * np.log(1 - predictions + 1e-8))
            epoch_loss += loss
            n_batches += 1

            # Backward pass with momentum
            error = predictions - batch_y
            dW = (1/batch_size) * np.dot(error, batch_X.T)
            db = (1/batch_size) * np.sum(error, axis=1, keepdims=True)

            # Update with momentum
            v_weights = momentum * v_weights - learning_rate * dW
            v_biases = momentum * v_biases - learning_rate * db
            
            model.weights += v_weights
            model.biases += v_biases

        # Evaluate on test set
        X_test_normalized = batch_normalize(X_test)
        test_predictions, _ = model.forward(X_test_normalized.T)
        test_predictions = np.argmax(test_predictions, axis=0)
        test_labels = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_predictions == test_labels) * 100

        # Print epoch statistics
        avg_epoch_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_weights = model.weights.copy()
            best_biases = model.biases.copy()
            print(f"New best accuracy: {best_accuracy:.2f}%")

    # Save best model
    save_dir = Path(__file__).parent / "saved_models"
    save_dir.mkdir(exist_ok=True)
    
    # Verify weights before saving
    print("\nVerifying best model before saving...")
    model.weights = best_weights
    model.biases = best_biases
    
    # Test on a few examples
    X_test_sample = batch_normalize(X_test[:5])
    predictions, _ = model.forward(X_test_sample.T)
    predicted_digits = np.argmax(predictions, axis=0)
    true_digits = np.argmax(y_test[:5], axis=1)
    
    print("Sample predictions before saving:")
    for i, (true, pred) in enumerate(zip(true_digits, predicted_digits)):
        print(f"Example {i+1}: True={true}, Predicted={pred}")
    
    # Save model with verified weights
    np.savez(save_dir / "single_layer_perceptron.npz", 
             weights=best_weights, 
             biases=best_biases)
    
    print(f"\nQuick training complete! Best accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to {save_dir / 'single_layer_perceptron.npz'}")
    print(f"Final weights shape: {best_weights.shape}")
    print(f"Final biases shape: {best_biases.shape}")


if __name__ == "__main__":
    main() 