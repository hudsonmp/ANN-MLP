import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class MNISTLoader:
    """A class for loading and preprocessing the MNIST dataset.
    
    This class provides functionality to:
    1. Load the MNIST dataset from OpenML
    2. Preprocess the data (normalization, feature selection)
    3. Convert to binary classification if needed
    4. Visualize the data
    
    The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9).
    Each image is 28x28 pixels, resulting in 784 features initially.
    """
    
    def __init__(
        self,
        subset_size: Optional[int] = None,
        random_state: int = 42,
        normalization: str = 'standard',
        use_feature_selection: bool = True,
        variance_threshold: float = 0.01,
        n_features_to_keep: Optional[int] = None
    ):
        """Initialize the MNIST loader.
        
        Args:
            subset_size: Number of samples to load (None for full dataset)
            random_state: Random seed for reproducibility
            normalization: Type of normalization to use ('standard' or 'minmax')
            use_feature_selection: Whether to use feature selection
            variance_threshold: Minimum variance threshold for features
            n_features_to_keep: Number of features to keep after selection
        """
        self.subset_size = subset_size
        self.random_state = random_state
        self.normalization = normalization
        self.use_feature_selection = use_feature_selection
        self.variance_threshold = variance_threshold
        self.n_features_to_keep = n_features_to_keep
        
        # Initialize preprocessing objects
        self.scaler = StandardScaler()
        
    def one_hot_encode(self, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """Convert integer labels to one-hot encoded format.
        
        Args:
            y: Integer labels of shape (n_samples,)
            num_classes: Number of classes
            
        Returns:
            One-hot encoded labels of shape (n_samples, num_classes)
        """
        return np.eye(num_classes)[y.astype(int)]
    
    def visualize_samples(self, X: np.ndarray, y: np.ndarray, num_samples: int = 5) -> None:
        """Visualize a few samples from the dataset.
        
        Args:
            X: Image data of shape (n_samples, 784)
            y: Labels
            num_samples: Number of samples to visualize
        """
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
        
        for i in range(num_samples):
            img = X[i].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {y[i]}')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
    
    def load_mnist(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess the MNIST dataset.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        print("Fetching MNIST dataset from OpenML...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        
        if self.subset_size is not None:
            indices = np.random.RandomState(self.random_state).permutation(len(X))[:self.subset_size]
            X = X[indices]
            y = y[indices]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Normalize the data
        if self.normalization == 'standard':
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        elif self.normalization == 'minmax':
            X_train = X_train / 255.0
            X_test = X_test / 255.0
        
        return X_train, y_train, X_test, y_test
    
    def prepare_binary_task(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        digit: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert the multi-class problem to binary classification.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            digit: The digit to use as the positive class
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test) for binary classification
        """
        # Convert labels to binary (1 for the specified digit, 0 for others)
        y_train_binary = (y_train.astype(int) == digit).astype(int)
        y_test_binary = (y_test.astype(int) == digit).astype(int)
        
        return X_train, y_train_binary, X_test, y_test_binary
