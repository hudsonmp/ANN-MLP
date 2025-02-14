import numpy as np
from sklearn import datasets
from models.mlp import MLP
from utils.data_preprocessing import normalize_features, train_test_split, one_hot_encode

def main():
    # Generate a toy dataset (moons dataset)
    X, y = datasets.make_moons(n_samples=100, noise=0.1)
    
    # Preprocess the data
    X = normalize_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and initialize the model
    layer_sizes = [2, 20, 2]  # Input dim: 2, Hidden layer: 20 neurons, Output dim: 2
    model = MLP(layer_sizes)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
