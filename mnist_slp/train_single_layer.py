import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from typing import Tuple
from mnist_slp.single_layer_perceptron.single_layer import SingleLayerForward
from mnist_slp.single_layer_perceptron.perceptron import Perceptron
from utils.mnist_loader import MNISTLoader
from utils.visualizer import plot_training_history, plot_confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler


def batch_normalize(X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Apply batch normalization to input data."""
    mean = np.mean(X, axis=0, keepdims=True)
    var = np.var(X, axis=0, keepdims=True)
    return (X - mean) / np.sqrt(var + epsilon)


def select_features(X_train, X_test, y_train, n_features=500):
    """Select the most informative features using mutual information."""
    # Convert one-hot encoded y to class labels
    y_train_labels = np.argmax(y_train, axis=1)
    
    # Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train_labels)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature indices
    selected_features = selector.get_support()
    
    return X_train_selected, X_test_selected, selected_features


def train_model(X_train, X_test, y_train, y_test, learning_rate, n_features=784):
    """Train model with specific learning rate and number of features."""
    # Select features if needed
    if n_features < 784:
        X_train, X_test, selected_features = select_features(X_train, X_test, y_train, n_features)
    
    # Create and configure model
    model = SingleLayerForward(n_features, 10)

    # Training parameters
    n_epochs = 150  # Increased epochs
    batch_size = 128  # Increased batch size
    momentum = 0.9
    
    # Learning rate decay
    decay_rate = 0.95
    decay_steps = 1000
    
    # Initialize momentum variables
    v_weights = np.zeros_like(model.weights)
    v_biases = np.zeros_like(model.biases)

    best_accuracy = 0
    patience = 7  # Increased patience
    patience_counter = 0
    
    # Training loop
    step = 0
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0
        n_batches = 0

        # Mini-batch training
        for i in range(0, len(X_train_shuffled), batch_size):
            # Decay learning rate
            current_lr = learning_rate * (decay_rate ** (step / decay_steps))
            step += 1
            
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
            v_weights = momentum * v_weights - current_lr * dW
            v_biases = momentum * v_biases - current_lr * db
            
            model.weights += v_weights
            model.biases += v_biases

        # Print epoch statistics
        avg_epoch_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}")

        # Evaluate on validation set
        X_test_normalized = batch_normalize(X_test)
        test_predictions, _ = model.forward(X_test_normalized.T)
        test_predictions = np.argmax(test_predictions, axis=0)
        test_labels = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_predictions == test_labels) * 100

        print(f"Validation Accuracy: {test_accuracy:.2f}%")

        # Early stopping
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            # Save best model
            save_dir = Path("saved_models")
            save_dir.mkdir(exist_ok=True)
            np.savez(save_dir / "single_layer_perceptron.npz", 
                    weights=model.weights, 
                    biases=model.biases)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered. Best accuracy: {best_accuracy:.2f}%")
                break

    return best_accuracy


def main() -> None:
    """
    Main function to experiment with different learning rates and feature selections.
    """
    # Load MNIST data
    print("Loading MNIST data...")
    mnist_loader = MNISTLoader(subset_size=None)  # Use full dataset
    X_train, X_test, y_train, y_test = mnist_loader.load_mnist()

    # Learning rates to try (exponential range)
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    # Feature counts to try
    feature_counts = [784, 500, 300]  # Full features, 500 features, 300 features

    best_accuracy = 0
    best_lr = None
    best_features = None

    # Grid search over learning rates and feature counts
    for lr in learning_rates:
        for n_features in feature_counts:
            print(f"\nTrying learning rate: {lr}, features: {n_features}")
            accuracy = train_model(X_train.copy(), X_test.copy(), 
                                 y_train.copy(), y_test.copy(), 
                                 learning_rate=lr, n_features=n_features)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_lr = lr
                best_features = n_features
                print(f"New best model! Accuracy: {best_accuracy:.2f}%")

    print("\nTraining complete!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Best learning rate: {best_lr}")
    print(f"Best feature count: {best_features}")


if __name__ == "__main__":
    main()
