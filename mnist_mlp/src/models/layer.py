import numpy as np
from typing import Tuple, Optional, Callable

class Layer:
    def __init__(self, input_dim: int, output_dim: int, activation_fn: Optional[Callable] = None):
        """Initialize a neural network layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of neurons in the layer
            activation_fn: Activation function to use (None for no activation)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        
        # Initialize weights and biases
        self.weights = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.biases = np.zeros((1, output_dim))
        
        # Cache for backpropagation
        self.input = None
        self.output = None
        self.activated_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        self.input = x
        self.output = np.dot(x, self.weights) + self.biases
        
        if self.activation_fn is not None:
            self.activated_output = self.activation_fn(self.output)
            return self.activated_output
        return self.output

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get layer parameters."""
        return self.weights, self.biases

    def set_params(self, weights: np.ndarray, biases: np.ndarray) -> None:
        """Set layer parameters."""
        self.weights = weights
        self.biases = biases
