import numpy as np
from mnist_mlp.mlp import MultiLayerPerceptron
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os
from typing import Tuple, List, Dict
import time

def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the preprocessed MNIST data with selected features."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed_data')
    
    try:
        # Load and ensure arrays are float32 for features and int32 for labels
        X_train = np.load(os.path.join(data_dir, 'X_train_selected.npy'), allow_pickle=True).astype(np.float32)
        X_test = np.load(os.path.join(data_dir, 'X_test_selected.npy'), allow_pickle=True).astype(np.float32)
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True).astype(np.int32)
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True).astype(np.int32)
        
        # Binarize the images using a threshold of 0.5
        X_train = (X_train > 0.5).astype(np.float32)
        X_test = (X_test > 0.5).astype(np.float32)
        
        # Verify data shapes
        print(f"Loaded data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        # Convert labels to one-hot encoding
        def to_one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
            return np.eye(num_classes)[y]
        
        y_train_one_hot = to_one_hot(y_train)
        y_test_one_hot = to_one_hot(y_test)
        
        # Verify no NaN values
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError("NaN values found in the data")
            
        return X_train, y_train_one_hot, X_test, y_test_one_hot, y_train, y_test
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def train_and_evaluate(
    epochs_list: List[int] = [100, 500, 1000],
    learning_rates: List[float] = [0.01, 0.001]
) -> List[Dict]:
    """Train the MLP model with different epochs and learning rates."""
    X_train, y_train_one_hot, X_test, y_test_one_hot, y_train, y_test = load_processed_data()
    
    # Create results and model directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    results = []
    total_configs = len(epochs_list) * len(learning_rates)
    config_count = 0
    
    for lr in learning_rates:
        for epochs in epochs_list:
            config_count += 1
            print(f"\nConfiguration {config_count}/{total_configs}")
            print(f"Training with learning_rate={lr}, epochs={epochs}")
            
            # Initialize model with larger architecture
            input_size = X_train.shape[1]
            hidden_size = 256  # Fixed larger hidden size
            
            try:
                model = MultiLayerPerceptron(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=10,
                    learning_rate=lr,
                    max_iterations=epochs,
                    batch_size=32,
                    random_state=42
                )
                
                # Train model with progress tracking
                start_time = time.time()
                model.fit(X_train, y_train_one_hot)
                training_time = time.time() - start_time
                
                # Evaluate on test set
                y_pred_proba = model.predict_proba(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"Training time: {training_time:.2f} seconds")
                print(f"Test Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred, digits=4))
                
                # Save model weights
                weights_data = {
                    'weights': model.weights,
                    'biases': model.biases,
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'output_size': 10
                }
                np.save('model/weights.npy', weights_data)
                print("Model weights saved successfully!")
                
                # Plot and save learning curve
                plt.figure(figsize=(10, 5))
                plt.plot(model.loss_history)
                plt.title(f'Learning Curve (lr={lr}, epochs={epochs})')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.savefig(os.path.join('results', f'learning_curve_lr{lr}_epochs{epochs}.png'))
                plt.close()
                
                results.append({
                    'learning_rate': lr,
                    'epochs': epochs,
                    'accuracy': accuracy,
                    'loss_history': model.loss_history,
                    'training_time': training_time
                })
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                continue
            
    return results

def main():
    """Main function to run the training and evaluation."""
    print("Starting MLP training and evaluation...")
    
    try:
        # Train with best configuration
        results = train_and_evaluate(
            epochs_list=[1000],
            learning_rates=[0.001]  # Using only the best learning rate
        )
        
        if not results:
            print("No successful training runs completed.")
            return
        
        # Get the result
        result = results[0]
        print("\nTraining Complete:")
        print(f"Learning Rate: {result['learning_rate']}")
        print(f"Epochs: {result['epochs']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Training Time: {result['training_time']:.2f} seconds")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 