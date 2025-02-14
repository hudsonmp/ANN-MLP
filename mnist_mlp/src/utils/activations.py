import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(x, 0)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU activation function."""
    return 1.0 * (x > 0)

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function."""
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
