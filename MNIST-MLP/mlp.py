import numpy as np
from typing import List, Tuple


class MultiLayerPerceptron:
    """A Multi-Layer Perceptron implementation for digit recognition.
    
    This class implements a neural network with multiple layers using
    the sigmoid activation function and backpropagation for training.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        random_state: int | None = None
    ) -> None:
        """Initialize the MLP.
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of training iterations
            random_state: Random seed for reproducibility
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Initialize weights with Xavier/Glorot initialization
            weight_scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * weight_scale
            )
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
            
        # Training history
        self.loss_history = []
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid activation function.
        
        Args:
            z: Input values
            
        Returns:
            Sigmoid activation of the inputs
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """Compute the derivative of the sigmoid function.
        
        Args:
            z: Input values
            
        Returns:
            Derivative of sigmoid for the inputs
        """
        sig = self._sigmoid(z)
        return sig * (1 - sig)
    
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Perform forward propagation through the network.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Tuple of (activations, weighted_sums) for each layer
        """
        current_activation = X
        activations = [X]
        weighted_sums = []
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(current_activation, w) + b
            weighted_sums.append(z)
            current_activation = self._sigmoid(z)
            activations.append(current_activation)
            
        return activations, weighted_sums
    
    def _backward(
        self,
        y: np.ndarray,
        activations: List[np.ndarray],
        weighted_sums: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Perform backpropagation to compute gradients.
        
        Args:
            y: True labels
            activations: List of activations from forward pass
            weighted_sums: List of weighted sums from forward pass
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = y.shape[0]
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Compute output layer error
        delta = activations[-1] - y
        
        # Backpropagate the error
        for l in reversed(range(len(self.weights))):
            weight_gradients[l] = np.dot(activations[l].T, delta) / m
            bias_gradients[l] = np.mean(delta, axis=0, keepdims=True)
            
            if l > 0:  # Not the first layer
                delta = np.dot(delta, self.weights[l].T) * self._sigmoid_derivative(weighted_sums[l-1])
                
        return weight_gradients, bias_gradients
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the binary cross-entropy loss.
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            
        Returns:
            Loss value
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiLayerPerceptron":
        """Train the neural network using gradient descent.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples, n_classes)
            
        Returns:
            self: The trained neural network
        """
        for iteration in range(self.max_iterations):
            # Forward pass
            activations, weighted_sums = self._forward(X)
            
            # Compute loss
            loss = self._compute_loss(activations[-1], y)
            self.loss_history.append(loss)
            
            # Backward pass
            weight_gradients, bias_gradients = self._backward(y, activations, weighted_sums)
            
            # Update parameters
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_gradients[i]
                self.biases[i] -= self.learning_rate * bias_gradients[i]
                
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input features.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities
        """
        activations, _ = self._forward(X)
        return activations[-1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input features.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int) 