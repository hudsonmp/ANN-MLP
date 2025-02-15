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
    """Test the saved model on MNIST test data."""
    print("Loading MNIST test data...")
    mnist_loader = MNISTLoader(subset_size=1000)  # Load a small subset for testing
    _, X_test, _, y_test = mnist_loader.load_mnist()

    print("\nLoading saved model...")
    model_path = Path(__file__).parent / "saved_models/single_layer_perceptron.npz"
    if not model_path.exists():
        raise RuntimeError("Model file not found!")

    model_params = np.load(str(model_path))
    print(f"Model parameters loaded. Keys: {model_params.files}")
    print(f"Weights shape: {model_params['weights'].shape}")
    print(f"Biases shape: {model_params['biases'].shape}")

    model = SingleLayerForward(784, 10)
    model.load_parameters(str(model_path))

    print("\nTesting on first 10 images...")
    for i in range(10):
        # Get a single test image
        image = X_test[i:i+1]
        true_label = np.argmax(y_test[i])

        # Apply the same preprocessing as in Django
        image_normalized = batch_normalize(image)
        image_normalized = image_normalized.T

        # Make prediction
        predictions, _ = model.forward(image_normalized)
        predicted_digit = np.argmax(predictions)
        confidence = float(predictions[predicted_digit])

        print(f"\nTest image {i+1}:")
        print(f"True digit: {true_label}")
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All probabilities: {predictions.ravel()}")


if __name__ == "__main__":
    main() 