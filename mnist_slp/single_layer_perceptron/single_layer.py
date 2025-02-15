import numpy as np
from typing import Tuple

class SingleLayerForward:
    """
    Handles forward propagation for a single layer neural network.
    Forward propagation is the process of:
    1. Taking input data
    2. Applying weights and biases
    3. Using an activation function to produce output/predictions
    """
    
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the layer with random weights and biases.
        
        Args:
            input_size: Number of input features
            output_size: Number of output classes/neurons
        """
        # Initialize weights using Xavier/Glorot initialization
        # This helps prevent the vanishing/exploding gradient problem
        limit = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
        self.biases = np.zeros((output_size, 1))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
        Squashes input to range [0,1], useful for binary classification
        """
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform forward propagation through the layer.
        
        Args:
            X: Input data of shape (n_features, n_samples)
        
        Returns:
            Tuple of:
                - Output after activation (predictions)
                - Pre-activation values (needed for backprop)
        """
        # Step 1: Linear transformation (z = Wx + b)
        # This is like fitting a line in linear regression
        z = np.dot(self.weights, X) + self.biases
        
        # Step 2: Apply activation function
        # This adds non-linearity, allowing the network to learn complex patterns
        a = self.sigmoid(z)
        
        return a, z

class SingleLayerBackward:
    """
    Handles backward propagation for a single layer neural network.
    Backward propagation is how the network learns by:
    1. Calculating prediction error
    2. Computing how each weight contributed to the error
    3. Updating weights using gradient descent to reduce future errors
    """
    
    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid function: f'(x) = f(x) * (1 - f(x))
        Used to calculate how much to adjust weights during backprop
        """
        sigmoid_z = 1 / (1 + np.exp(-z))
        return sigmoid_z * (1 - sigmoid_z)
    
    def backward(self, X: np.ndarray, Y: np.ndarray, A: np.ndarray, Z: np.ndarray, 
                weights: np.ndarray, learning_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform backward propagation to update weights and biases.
        
        Args:
            X: Input data of shape (n_features, n_samples)
            Y: True labels of shape (n_classes, n_samples)
            A: Output from forward propagation (after activation)
            Z: Pre-activation values from forward propagation
            weights: Current weights of the layer
            learning_rate: How fast to update weights (small steps)
        
        Returns:
            Tuple of:
                - Updated weights
                - Updated biases
        """
        m = X.shape[1]  # Number of training examples
        
        # Step 1: Calculate the error
        # Error = prediction - actual
        # This tells us how far off our predictions were
        error = A - Y
        
        # Step 2: Calculate gradient of loss with respect to Z
        # This tells us how the loss changes with small changes in Z
        # We multiply by sigmoid derivative due to chain rule
        dZ = error * self.sigmoid_derivative(Z)
        
        # Step 3: Calculate gradients for weights and biases
        # This tells us how to adjust weights and biases to reduce error
        dW = (1/m) * np.dot(dZ, X.T)  # Average gradient across all examples
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Step 4: Update weights and biases using gradient descent
        # Take small steps (learning_rate) in direction that reduces error
        weights = weights - learning_rate * dW
        biases = weights - learning_rate * db
        
        return weights, biases

# Example usage:
def train_step(X: np.ndarray, Y: np.ndarray, 
               forward_layer: SingleLayerForward, 
               backward_layer: SingleLayerBackward,
               learning_rate: float = 0.01) -> None:
    """
    Perform one training step (forward + backward pass).
    
    Args:
        X: Input data
        Y: True labels
        forward_layer: Instance of SingleLayerForward
        backward_layer: Instance of SingleLayerBackward
        learning_rate: Learning rate for gradient descent
    """
    # Forward pass: make predictions
    A, Z = forward_layer.forward(X)
    
    # Backward pass: learn from errors
    new_weights, new_biases = backward_layer.backward(
        X, Y, A, Z, forward_layer.weights, learning_rate
    )
    
    # Update the weights and biases in the forward layer
    forward_layer.weights = new_weights
    forward_layer.biases = new_biases 