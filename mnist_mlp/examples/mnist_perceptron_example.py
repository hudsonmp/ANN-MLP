import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.mnist_loader import load_mnist, prepare_binary_task
from src.utils.visualizer import (
    plot_digit, plot_multiple_digits, 
    plot_training_history, plot_confusion_matrix
)
from src.models.perceptron import Perceptron

def main():
    # Load MNIST dataset (using a subset for faster training)
    X_train, X_test, y_train, y_test = load_mnist(subset_size=10000)
    
    # Convert to binary classification (detect digit 0 vs non-0)
    X_train, X_test, y_train_binary, y_test_binary = prepare_binary_task(
        X_train, X_test, y_train, y_test, digit=0
    )
    
    # Show some example digits
    print("\nShowing some example digits from the dataset:")
    plot_multiple_digits(X_train[:5], y_train[:5])
    
    # Create and train the perceptron
    print("\nTraining perceptron...")
    perceptron = Perceptron(
        n_features=784,  # MNIST images are 28x28 = 784 features
        learning_rate=0.01,
        max_iterations=100,
        model_type='logistic',
        random_state=42
    )
    
    # Train the model
    perceptron.fit(X_train, y_train_binary)
    
    # Plot training history
    print("\nPlotting training history:")
    plot_training_history(perceptron.loss_history)
    
    # Make predictions
    y_pred = perceptron.predict_classes(X_test)
    
    # Show confusion matrix and classification report
    print("\nModel evaluation:")
    plot_confusion_matrix(y_test_binary, y_pred)
    
    # Show some example predictions
    print("\nShowing some example predictions:")
    test_indices = range(5)  # Show first 5 test examples
    plot_multiple_digits(
        X_test[test_indices],
        labels=y_test[test_indices],
        predictions=y_pred[test_indices]
    )

if __name__ == "__main__":
    main() 