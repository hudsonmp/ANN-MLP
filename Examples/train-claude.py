# train.py
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from mnist_mlp.mlp import MultiLayerPerceptron as MLP

def load_mnist():
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0  # Normalize pixel values
    y = y.astype(np.int32)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, X_test, y_test, 
                epochs=10, batch_size=32, learning_rate=0.01):
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        total_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass
            output = model.forward(X_batch)
            
            # Compute cross-entropy loss
            y_one_hot = np.zeros((batch_size, 10))
            y_one_hot[np.arange(batch_size), y_batch] = 1
            loss = -np.sum(y_one_hot * np.log(output + 1e-7)) / batch_size
            total_loss += loss
            
            # Backward pass
            model.backward(X_batch, y_batch, learning_rate)
        
        # Evaluate on test set
        test_output = model.forward(X_test)
        predictions = np.argmax(test_output, axis=1)
        accuracy = np.mean(predictions == y_test)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average Loss: {total_loss/n_batches:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("-" * 30)
    
    return model

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_mnist()
    
    # Initialize model
    model = MLP(input_size=784, hidden_size=128, output_size=10)
    
    # Train model
    model = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=10, batch_size=32, learning_rate=0.01
    )
    
    # Save trained model
    model.save_weights('model/weights.npy')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()