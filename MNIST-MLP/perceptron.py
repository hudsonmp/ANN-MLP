import numpy as np
from typing import Literal


class Perceptron:
    """A basic perceptron implementation supporting both linear and logistic regression.

    This class implements a single-layer perceptron that can be used for binary classification
    using either linear or logistic regression with gradient descent optimization.
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        model_type: Literal["linear", "logistic"] = "logistic",
        random_state: int | None = None,
    ) -> None:
        """Initialize the perceptron.

        Args:
            n_features: Number of input features
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of training iterations
            model_type: Type of regression model ('linear' or 'logistic')
            random_state: Random seed for reproducibility
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.model_type = model_type

        # Initialize weights and bias
        self.weights = (
            np.random.randn(n_features) * 0.01
        )  # Initialize weights with small random values
        self.bias = 0.0  # Initialize bias term separately

        # Training history
        self.loss_history = []

    def linear(self, inputs: np.ndarray) -> np.ndarray:
        """Compute linear activation.

        Args:
            inputs: Input values

        Returns:
            Linear activation of the inputs
        """
        Z = np.dot(inputs, self.weights) + self.bias
        return Z

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid function.

        Args:
            z: Input values

        Returns:
            Sigmoid of the input values
        """
        # Clip values to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _step(self, z: np.ndarray) -> np.ndarray:
        """Compute the Heaviside step function.

        Args:
            z: Input values

        Returns:
            Binary step output (0 or 1) for each input value
        """
        if self.model_type == "linear":
            return z
        else:
            return np.where(z >= 0, 1, 0)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward pass.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Model predictions
        """
        z = np.dot(X, self.weights) + self.bias
        if self.model_type == "logistic":
            return self._sigmoid(z)
        return z

    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the loss function.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Loss value
        """
        if self.model_type == "logistic":
            # Binary cross-entropy loss
            epsilon = 1e-15  # Small constant to avoid log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Mean squared error loss
            return np.mean((y_pred - y_true) ** 2) / 2

    def _compute_gradients(
        self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Compute gradients for weights and bias.

        Args:
            X: Input features
            y_pred: Predicted values
            y_true: True values

        Returns:
            Tuple of (weight gradients, bias gradient)
        """
        m = X.shape[0]  # Number of training examples
        error = y_pred - y_true  # Difference between predictions and actual values

        # Same formula for both linear and logistic regression
        dw = np.dot(X.T, error) / m  # Gradient for weights
        db = np.mean(error)  # Gradient for bias

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Train the perceptron using gradient descent.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            self: The trained perceptron
        """
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred = self._forward(X)

            # Compute loss
            loss = self._compute_loss(y_pred, y)
            self.loss_history.append(loss)

            # Compute gradients
            dw, db = self._compute_gradients(X, y_pred, y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Make predictions for input features.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Predictions (probabilities for logistic regression, real values for linear regression)
        """
        return self._forward(Z)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input features.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Class predictions (0 or 1)
        """
        predictions = self.predict(X)
        if self.model_type == "logistic":
            return (predictions >= 0.5).astype(int)
        else:
            return (predictions >= 0.0).astype(int)
