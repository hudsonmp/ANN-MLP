import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from typing import Tuple, List
from .mnist_loader import load_mnist_data
from utils.feature_selection import BackwardFeatureSelector
import matplotlib.pyplot as plt
import seaborn as sns


class BackwardFeatureSelector:
    def __init__(self, variance_threshold: float = 0.01, cv_folds: int = 3):
        """
        Initialize the backward feature selector.
        
        Args:
            variance_threshold (float): Minimum variance threshold for features
            cv_folds (int): Number of cross-validation folds for feature evaluation
        """
        self.variance_threshold = variance_threshold
        self.cv_folds = cv_folds
        self.selected_features = None
        self.feature_variances = None
    
    def calculate_feature_variances(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate variance for each feature across all samples.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Variance of each feature
        """
        return np.var(X, axis=0)
    
    def remove_low_variance_features(self, X: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Remove features with variance below the threshold.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            Tuple[np.ndarray, List[int]]: Filtered data and indices of selected features
        """
        self.feature_variances = self.calculate_feature_variances(X)
        self.selected_features = np.where(self.feature_variances >= self.variance_threshold)[0]
        return X[:, self.selected_features], self.selected_features.tolist()
    
    def evaluate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate feature importance using logistic regression and cross-validation.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            y (np.ndarray): Target labels
            
        Returns:
            np.ndarray: Feature importance scores
        """
        model = LogisticRegression(max_iter=1000)
        base_score = np.mean(cross_val_score(model, X, y, cv=self.cv_folds))
        importance_scores = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            X_without_feature = np.delete(X, i, axis=1)
            score = np.mean(cross_val_score(model, X_without_feature, y, cv=self.cv_folds))
            importance_scores[i] = base_score - score
            
        return importance_scores
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       n_features_to_keep: int = None) -> Tuple[np.ndarray, List[int]]:
        """
        Perform backward feature selection.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            y (np.ndarray): Target labels
            n_features_to_keep (int, optional): Number of features to keep
            
        Returns:
            Tuple[np.ndarray, List[int]]: Selected features and their indices
        """
        # First remove low variance features
        X_filtered, selected_indices = self.remove_low_variance_features(X)
        
        if n_features_to_keep is None:
            return X_filtered, selected_indices
        
        # Calculate importance scores for remaining features
        importance_scores = self.evaluate_feature_importance(X_filtered, y)
        
        # Select top n_features_to_keep based on importance scores
        top_features = np.argsort(importance_scores)[-n_features_to_keep:]
        final_selected_features = [selected_indices[i] for i in top_features]
        
        return X_filtered[:, top_features], final_selected_features

    def create_pixel_intensity_heatmap(self, X: np.ndarray, save_path: str = None) -> np.ndarray:
        """
        Create a heatmap showing the average pixel intensity across all samples.
        Also returns the matrix of average intensities.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, 784)
            save_path (str, optional): Path to save the heatmap image
            
        Returns:
            np.ndarray: Matrix of average pixel intensities (28x28)
        """
        # Reshape to (n_samples, 28, 28) if needed
        if X.shape[1] == 784:
            X_reshaped = X.reshape(-1, 28, 28)
        else:
            X_reshaped = X
            
        # Calculate average intensity for each pixel
        avg_intensities = np.mean(X_reshaped, axis=0)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_intensities, cmap='viridis', 
                   xticklabels=False, yticklabels=False)
        plt.title('Average Pixel Intensity Across All Samples')
        plt.colorbar(label='Average Intensity')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return avg_intensities
    
    def analyze_low_activity_pixels(self, X: np.ndarray, 
                                  intensity_threshold: float = 0.05,
                                  occurrence_threshold: float = 0.02) -> Tuple[List[int], np.ndarray]:
        """
        Identify pixels that are consistently below a certain intensity threshold.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, 784)
            intensity_threshold (float): Threshold for considering a pixel as "low intensity"
            occurrence_threshold (float): Fraction of samples where pixel must be low intensity
            
        Returns:
            Tuple[List[int], np.ndarray]: List of indices to remove and binary mask of kept pixels
        """
        # Reshape if needed
        if X.shape[1] == 784:
            X_reshaped = X.reshape(-1, 28, 28)
        else:
            X_reshaped = X
            
        # Count how often each pixel is below the threshold
        low_intensity_mask = X_reshaped < intensity_threshold
        low_intensity_counts = np.mean(low_intensity_mask, axis=0)
        
        # Find pixels that are consistently low intensity
        pixels_to_remove_mask = low_intensity_counts > (1 - occurrence_threshold)
        pixels_to_remove = np.where(pixels_to_remove_mask.flatten())[0]
        
        return pixels_to_remove.tolist(), ~pixels_to_remove_mask

def main():
    """
    Example usage of the BackwardFeatureSelector with visualization.
    """
    # Load MNIST data
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Initialize feature selector
    selector = BackwardFeatureSelector(variance_threshold=0.01)
    
    # Create and save pixel intensity heatmap
    avg_intensities = selector.create_pixel_intensity_heatmap(
        X_train, save_path='mnist_mlp/pixel_intensity_heatmap.png'
    )
    
    # Analyze low activity pixels
    pixels_to_remove, keep_mask = selector.analyze_low_activity_pixels(
        X_train, intensity_threshold=0.05, occurrence_threshold=0.02
    )
    
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Number of consistently low-intensity pixels: {len(pixels_to_remove)}")
    print(f"Pixels to remove (first 10): {pixels_to_remove[:10]}...")
    
    # Remove low activity pixels
    X_filtered = np.delete(X_train, pixels_to_remove, axis=1)
    
    # Perform additional feature selection if desired
    remaining_features_to_keep = 392 - len(pixels_to_remove)
    if remaining_features_to_keep > 0:
        X_selected, selected_features = selector.select_features(
            X_filtered, y_train, n_features_to_keep=remaining_features_to_keep
        )
        
        print(f"\nAfter variance-based selection:")
        print(f"Number of selected features: {len(selected_features)}")
        print(f"Selected feature indices: {selected_features[:10]}...")
    
    # Calculate and print variance statistics
    print(f"\nVariance statistics of original features:")
    print(f"Mean variance: {np.mean(selector.feature_variances):.4f}")
    print(f"Max variance: {np.max(selector.feature_variances):.4f}")
    print(f"Min variance: {np.min(selector.feature_variances):.4f}")

if __name__ == "__main__":
    main() 