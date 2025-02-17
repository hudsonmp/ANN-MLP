import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import KFold
from .feature_selection import BackwardFeatureSelector


class MultiLayerPerceptron:
    """A Multi-Layer Perceptron implementation for digit recognition.
    
    This class implements a neural network with multiple layers using
    the ReLU activation function for hidden layers and softmax for the output layer.
    ReLU is used instead of sigmoid because:
    1. It's more suitable for computer-drawn digits where pressure sensitivity isn't relevant
    2. It helps prevent vanishing gradients
    3. It's computationally more efficient
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128, 
        output_size: int = 10,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        max_iterations: int = 1000,
        random_state: int | None = None
    ) -> None:
        if random_state is not None:
            np.random.seed(random_state)
            
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize lists to store weights and biases
        self.weights = []
        self.biases = []
        
        # Layer sizes for initialization
        layer_sizes = [input_size, hidden_size, output_size]
        
        # He initialization for better gradient flow
        for i in range(len(layer_sizes) - 1):
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            )
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
        
        # Training history
        self.loss_history = []
        
    def _relu(self, z: np.ndarray) -> np.ndarray:
        """Compute the ReLU activation function.
        
        ReLU(x) = max(0, x)
        
        Args:
            z: Input values
            
        Returns:
            ReLU activation of the inputs
        """
        return np.maximum(0, z)
    
    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Compute the derivative of the ReLU function.
        
        ReLU'(x) = 1 if x > 0 else 0
        
        Args:
            z: Input values
            
        Returns:
            Derivative of ReLU for the inputs
        """
        return (z > 0).astype(float)
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Compute the softmax activation function.
        
        Used in the output layer for multi-class classification.
        
        Args:
            z: Input values
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Perform forward propagation through the network.
        
        Uses ReLU activation for hidden layers and softmax for output layer.
        
        Args:
            X: Input features of shape (batch_size, n_features)
            
        Returns:
            Tuple of (activations, weighted_sums) for each layer
        """
        current_activation = X
        activations = [X]  # List to store all activations, including input
        weighted_sums = []  # List to store all z values
        
        # Process all layers except the last one
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            weighted_sums.append(z)
            current_activation = self._relu(z)  # ReLU for hidden layers
            activations.append(current_activation)
        
        # Process the output layer with softmax
        z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        weighted_sums.append(z)
        current_activation = self._softmax(z)  # Softmax for output layer
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
            y: True labels (one-hot encoded)
            activations: List of activations from forward pass
            weighted_sums: List of weighted sums from forward pass
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = y.shape[0]  # Batch size
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Compute output layer error (cross-entropy with softmax derivative simplifies to this)
        delta = activations[-1] - y  # Shape: (batch_size, output_size)
        
        # Backpropagate the error
        for l in reversed(range(len(self.weights))):
            # Compute gradients for current layer
            weight_gradients[l] = np.dot(activations[l].T, delta) / m
            bias_gradients[l] = np.mean(delta, axis=0, keepdims=True)
            
            if l > 0:  # Not the first layer
                # Compute error for previous layer
                delta = np.dot(delta, self.weights[l].T) * self._relu_derivative(weighted_sums[l-1])
                
        return weight_gradients, bias_gradients
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the categorical cross-entropy loss.
        
        J(θ) = -(1/m) * Σ Σ y_ij * log(ŷ_ij)
        where m is the batch size, y_ij is the true probability of class j for example i,
        and ŷ_ij is the predicted probability.
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels (one-hot encoded)
            
        Returns:
            Loss value
        """
        m = y_true.shape[0]  # Batch size
        epsilon = 1e-15  # Small constant to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / m
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiLayerPerceptron":
        """Train the neural network using mini-batch gradient descent.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples, n_classes)
            
        Returns:
            self: The trained neural network
        """
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size  # Ceiling division
        
        for iteration in range(self.max_iterations):
            # Shuffle the training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            
            # Process mini-batches
            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                # Get current batch
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                activations, weighted_sums = self._forward(X_batch)
                
                # Compute loss for this batch
                batch_loss = self._compute_loss(activations[-1], y_batch)
                total_loss += batch_loss
                
                # Backward pass
                weight_gradients, bias_gradients = self._backward(y_batch, activations, weighted_sums)
                
                # Update parameters
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * weight_gradients[i]
                    self.biases[i] -= self.learning_rate * bias_gradients[i]
            
            # Record average loss for this epoch
            avg_loss = total_loss / n_batches
            self.loss_history.append(avg_loss)
                
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
        return np.argmax(probas, axis=1)
    
    @staticmethod
    def perform_feature_selection(
        X: np.ndarray,
        y: np.ndarray,
        variance_threshold: float = 0.01,
        n_features_to_keep: int | None = None
    ) -> Tuple[np.ndarray, List[int], int]:
        """Perform feature selection using BackwardFeatureSelector.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            y: Target labels
            variance_threshold: Minimum variance threshold for features
            n_features_to_keep: Number of features to keep after selection
            
        Returns:
            Tuple of (selected features array, indices of selected features, number of selected features)
        """
        selector = BackwardFeatureSelector(variance_threshold=variance_threshold)
        return selector.select_features(X, y, n_features_to_keep)
    
    @classmethod
    def cross_validate_hyperparameters(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict,
        n_splits: int = 5,
        random_state: int | None = None
    ) -> Tuple[Dict, float]:
        """Perform cross-validation to find optimal hyperparameters.
        
        Args:
            X: Input features
            y: Target labels
            param_grid: Dictionary of parameters to try
                {
                    'hidden_size': [64, 128, 256],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'batch_size': [16, 32, 64]
                }
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (best parameters, best score)
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        best_score = -np.inf
        best_params = None
        
        # Generate all parameter combinations
        from itertools import product
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in product(*param_grid.values())]
        
        for params in param_combinations:
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create and train model with current parameters
                model = cls(
                    input_size=X.shape[1],
                    hidden_size=params['hidden_size'],
                    learning_rate=params['learning_rate'],
                    batch_size=params['batch_size'],
                    random_state=random_state
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred = model.predict_proba(X_val)
                score = model._compute_accuracy(y_pred, y_val)
                scores.append(score)
            
            # Calculate mean score for this parameter combination
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        return best_params, best_score
    
    def _compute_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute classification accuracy.
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels (one-hot encoded)
            
        Returns:
            Classification accuracy
        """
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)) 