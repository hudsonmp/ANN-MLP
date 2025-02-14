import numpy as np
from typing import List, Tuple, Optional
from .layer import Layer
from ..utils.activations import relu, softmax

class MLP:
    def __init__(self, layer_sizes: List[int], hidden_activation=relu, output_activation=softmax):
        """Initialize Multi-Layer Perceptron.
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
            hidden_activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            is_output_layer = i == len(layer_sizes) - 2
            activation = output_activation if is_output_layer else hidden_activation
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activation)
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        current_output = x
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions for the input data."""
        output = self.forward(x)
        return np.argmax(output, axis=1)

    def get_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all network parameters."""
        return [layer.get_params() for layer in self.layers]

    def set_params(self, params: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Set all network parameters."""
        for layer, (weights, biases) in zip(self.layers, params):
            layer.set_params(weights, biases)
