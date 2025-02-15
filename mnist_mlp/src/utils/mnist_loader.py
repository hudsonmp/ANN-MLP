import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist(subset_size=None, random_state=42):
    """Load MNIST dataset using OpenML.
    
    The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9).
    Each image is 28x28 pixels, resulting in 784 features.
    
    Dataset source:
    - OpenML: https://www.openml.org/d/554
    
    Args:
        subset_size: Number of samples to load (None for full dataset)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) where X is the data and y is labels
    """
    print("Loading MNIST dataset from OpenML...")
    
    # Load from OpenML
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    if subset_size is not None:
        # Use only a subset of the data if specified
        X = X[:subset_size]
        y = y[:subset_size]
        
    # Split the data into training and testing sets (60k train, 10k test to match original)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=random_state
    )

    # Convert data to float32 for better memory usage
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    #pixel values range from 0 to 255
    #Must be floats to normalize and train the model
    
    # Normalize pixel values to [0, 1]
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Normalize pixel values to [0, 1]
    X_train /= 255.0
    X_test /= 255.0
    
    # Convert labels to integers
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    #labels range from 0 to 9
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Image size: 28x28 pixels ({X_train.shape[1]} features)")
    print(f"Classes: Digits 0-9")
    
    return X_train, X_test, y_train, y_test

def prepare_binary_task(X_train, X_test, y_train, y_test, digit=0):
    """Convert MNIST to binary classification task.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        digit: The digit to classify (vs all other digits)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) with binary labels
    """
    # Convert to binary classification (digit vs non-digit)
    y_train_binary = (y_train == digit).astype(int)
    y_test_binary = (y_test == digit).astype(int)
    
    print(f"Converted to binary classification task: digit {digit} vs non-{digit}")
    print(f"Positive class (digit {digit}): {sum(y_train_binary)} training samples, {sum(y_test_binary)} test samples")
    
    return X_train, X_test, y_train_binary, y_test_binary 