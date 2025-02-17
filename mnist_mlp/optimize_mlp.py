import numpy as np
from .mlp import MultiLayerPerceptron
from .mnist_loader import load_mnist_data

def optimize_mlp_model():
    """Optimize the MLP model using cross-validation and feature selection."""
    # Load MNIST data
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Perform feature selection
    print('Performing feature selection...')
    X_train_selected, selected_features = MultiLayerPerceptron.perform_feature_selection(
        X_train, 
        y_train,
        variance_threshold=0.01,
        n_features_to_keep=392  # Reduce to ~half the original features
    )
    
    X_test_selected = X_test[:, selected_features]
    
    print(f'Selected {X_train_selected.shape[1]} features from {X_train.shape[1]} original features')
    
    # Define parameter grid for cross-validation
    param_grid = {
        'hidden_size': [64, 128, 256],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64]
    }
    
    # Perform cross-validation
    print('\nPerforming cross-validation to find optimal hyperparameters...')
    best_params, best_score = MultiLayerPerceptron.cross_validate_hyperparameters(
        X_train_selected,
        y_train,
        param_grid,
        n_splits=5,
        random_state=42
    )
    
    print('\nBest parameters found:')
    for param, value in best_params.items():
        print(f'{param}: {value}')
    print(f'Best cross-validation accuracy: {best_score:.4f}')
    
    # Train final model with best parameters
    print('\nTraining final model with best parameters...')
    final_model = MultiLayerPerceptron(
        input_size=X_train_selected.shape[1],
        hidden_size=best_params['hidden_size'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        random_state=42
    )
    
    final_model.fit(X_train_selected, y_train)
    
    # Evaluate on test set
    test_predictions = final_model.predict_proba(X_test_selected)
    test_accuracy = final_model._compute_accuracy(test_predictions, y_test)
    print(f'\nFinal test accuracy: {test_accuracy:.4f}')
    
    return final_model, selected_features

if __name__ == '__main__':
    optimize_mlp_model() 