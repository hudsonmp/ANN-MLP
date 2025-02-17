import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from utils.feature_selection import BackwardFeatureSelector
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
        self.feature_selector = BackwardFeatureSelector(
            variance_threshold=variance_threshold,
            cv_folds=3
        )
        
        # Store selected features for test set transformation
        self.selected_features = None
    
    def one_hot_encode(self, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """Convert labels to one-hot encoded format.
        
        Args:
            y: Input labels of shape (n_samples,)
            num_classes: Number of unique classes
            
        Returns:
            One-hot encoded labels of shape (n_samples, num_classes)
        """
        return np.eye(num_classes)[y.astype(int)]
    
    def visualize_samples(self, X: np.ndarray, y: np.ndarray, num_samples: int = 5) -> None:
        """Visualize sample images from the dataset.
        
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
        
        This method:
        1. Loads raw data from OpenML
        2. Splits into train/test sets
        3. Applies normalization
        4. Optionally performs feature selection
        5. Converts labels to one-hot encoding
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) where:
                X_train, X_test: Preprocessed feature matrices
                y_train, y_test: One-hot encoded labels
        """
        print("Loading MNIST dataset from OpenML...")
        
        # Load raw data
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        
        if self.subset_size is not None:
            # Use only a subset of the data if specified
            X = X[:self.subset_size]
            y = y[:self.subset_size]
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Convert to float32 for better memory usage
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        
        # Normalize the data
        if self.normalization == 'standard':
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:  # minmax normalization to [0, 1]
            X_train /= 255.0
            X_test /= 255.0
        
        # Perform feature selection if enabled
        if self.use_feature_selection:
            print("\nPerforming feature selection...")
            # First visualize pixel intensity heatmap
            self.feature_selector.create_pixel_intensity_heatmap(X_train)
            
            # Select features
            X_train, self.selected_features = self.feature_selector.select_features(
                X_train, y_train.astype(int), self.n_features_to_keep
            )
            # Apply same feature selection to test set
            X_test = X_test[:, self.selected_features]
            
            print(f"Selected {len(self.selected_features)} features")
        
        # Convert labels to integers and then one-hot encode
        y_train = y_train.astype('int32')
        y_test = y_test.astype('int32')
        y_train = self.one_hot_encode(y_train)
        y_test = self.one_hot_encode(y_test)
        
        print(f"\nDataset loaded and preprocessed:")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features per sample: {X_train.shape[1]}")
        print(f"Classes: Digits 0-9")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_binary_task(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        digit: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert MNIST to a binary classification task.
        
        This method converts the multi-class MNIST dataset into a binary
        classification problem where the task is to identify a specific digit
        versus all other digits.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels (one-hot encoded)
            y_test: Test labels (one-hot encoded)
            digit: The digit to classify (vs all other digits)
            
        Returns:
            tuple: (X_train, X_test, y_train_binary, y_test_binary)
                where the labels are now binary (0 or 1)
        """
        # Convert one-hot encoded labels back to integers
        y_train_int = np.argmax(y_train, axis=1)
        y_test_int = np.argmax(y_test, axis=1)
        
        # Convert to binary classification (digit vs non-digit)
        y_train_binary = (y_train_int == digit).astype(int)
        y_test_binary = (y_test_int == digit).astype(int)
        
        print(f"\nConverted to binary classification task:")
        print(f"Target digit: {digit}")
        print(f"Positive samples in training: {np.sum(y_train_binary)}")
        print(f"Positive samples in test: {np.sum(y_test_binary)}")
        
        return X_train, X_test, y_train_binary, y_test_binary
