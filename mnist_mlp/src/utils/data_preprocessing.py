import numpy as np
from typing import Tuple

def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features to have zero mean and unit variance."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert labels to one-hot encoded format."""
    return np.eye(num_classes)[y]

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and test sets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    test_size = int(test_size * num_samples)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test
