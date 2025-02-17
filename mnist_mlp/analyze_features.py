import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import MNISTLoader
from feature_selection import BackwardFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def analyze_mnist_features():
    """Analyze MNIST features to determine optimal feature count."""
    # Initialize and load MNIST data
    print("Loading MNIST data...")
    loader = MNISTLoader(
        normalization='standard',
        use_feature_selection=False  # We'll do feature selection manually
    )
    X_train, y_train, X_test, y_test = loader.load_mnist()
    
    # Convert labels to one-hot encoding
    y_train = loader.one_hot_encode(y_train)
    y_test = loader.one_hot_encode(y_test)
    
    print(f"Original feature shape: {X_train.shape}")
    
    # Initialize feature selector
    selector = BackwardFeatureSelector(variance_threshold=0.01)
    
    # First, analyze pixel intensity distribution
    print("\nAnalyzing pixel intensity distribution...")
    avg_intensities = selector.create_pixel_intensity_heatmap(
        X_train, 
        save_path="pixel_intensity_heatmap.png"
    )
    
    # Analyze low activity pixels
    print("\nAnalyzing low activity pixels...")
    pixels_to_remove, keep_mask = selector.analyze_low_activity_pixels(
        X_train,
        intensity_threshold=0.05,
        occurrence_threshold=0.02
    )
    
    print(f"Number of consistently low-intensity pixels: {len(pixels_to_remove)}")
    
    # Remove low activity pixels
    X_train_filtered = np.delete(X_train, pixels_to_remove, axis=1)
    print(f"Shape after removing low activity pixels: {X_train_filtered.shape}")
    
    # Try different feature counts to find optimal number
    feature_counts = [100, 200, 300, 400, 500, 600, 700]
    accuracies = []
    
    print("\nTesting different feature counts...")
    for n_features in feature_counts:
        print(f"Testing with {n_features} features...")
        X_selected, _ = selector.select_features(
            X_train_filtered, 
            y_train,
            n_features_to_keep=n_features
        )
        
        # Use logistic regression to evaluate feature set
        model = LogisticRegression(max_iter=1000)
        scores = cross_val_score(model, X_selected, np.argmax(y_train, axis=1), cv=3)
        accuracies.append(np.mean(scores))
        
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, accuracies, 'o-')
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Feature Count vs Model Performance')
    plt.grid(True)
    plt.savefig('feature_selection_analysis.png')
    plt.close()
    
    # Find optimal feature count
    optimal_idx = np.argmax(accuracies)
    optimal_features = feature_counts[optimal_idx]
    
    print("\nResults:")
    print(f"Optimal number of features: {optimal_features}")
    print(f"Best accuracy: {accuracies[optimal_idx]:.4f}")
    
    # Get final selected features with optimal count
    X_final, selected_features = selector.select_features(
        X_train_filtered,
        y_train,
        n_features_to_keep=optimal_features
    )
    
    return optimal_features, selected_features, pixels_to_remove

if __name__ == '__main__':
    optimal_features, selected_features, pixels_to_remove = analyze_mnist_features() 