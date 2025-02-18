import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from typing import Tuple, List
import os


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

    def remove_low_variance_features(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Remove features with variance below the threshold.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            Tuple[np.ndarray, List[int]]: Filtered data and indices of selected features
        """
        self.feature_variances = self.calculate_feature_variances(X)
        self.selected_features = np.where(
            self.feature_variances >= self.variance_threshold
        )[0]
        return X[:, self.selected_features], self.selected_features.tolist()

    def evaluate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate feature importance using logistic regression coefficients.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            y (np.ndarray): Target labels

        Returns:
            np.ndarray: Feature importance scores
        """
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # For multi-class problems, take the mean absolute coefficient values across all classes
        importance_scores = np.mean(np.abs(model.coef_), axis=0)
        return importance_scores

    def select_features(
        self, X: np.ndarray, y: np.ndarray, n_features_to_keep: int = None
    ) -> Tuple[np.ndarray, List[int], int]:
        """
        Perform backward feature selection.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            y (np.ndarray): Target labels
            n_features_to_keep (int, optional): Number of features to keep

        Returns:
            Tuple[np.ndarray, List[int], int]: Selected features, their indices, and number of selected features
        """
        # First remove low variance features
        X_filtered, selected_indices = self.remove_low_variance_features(X)

        if n_features_to_keep is None:
            return X_filtered, selected_indices, len(selected_indices)

        # Calculate importance scores for remaining features
        importance_scores = self.evaluate_feature_importance(X_filtered, y)

        # Select top n_features_to_keep based on importance scores
        top_features = np.argsort(importance_scores)[-n_features_to_keep:]
        final_selected_features = [selected_indices[i] for i in top_features]

        return X_filtered[:, top_features], final_selected_features, n_features_to_keep

    def analyze_low_activity_pixels(
        self,
        X: np.ndarray,
        intensity_threshold: float = 0.05,
        occurrence_threshold: float = 0.02,
    ) -> Tuple[List[int], np.ndarray]:
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

    def save_processed_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        selected_features: List[int],
        save_dir: str = 'processed_data'
    ) -> None:
        """
        Save the processed datasets to files.
        
        Args:
            X_train: Selected training features
            X_test: Selected test features
            y_train: Training labels
            y_test: Test labels
            selected_features: Indices of selected features
            save_dir: Directory to save the processed data
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the transformed datasets
        np.save(os.path.join(save_dir, 'X_train_selected.npy'), X_train)
        np.save(os.path.join(save_dir, 'X_test_selected.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
        np.save(os.path.join(save_dir, 'selected_features.npy'), np.array(selected_features))
        
        print(f"\nProcessed data saved to '{save_dir}' directory")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")


def main():
    """
    Example usage of the BackwardFeatureSelector.
    """
    # Load MNIST data directly from OpenML
    print("Loading MNIST dataset...")
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0  # Normalize pixel values
    y = y.astype(np.int32)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize feature selector with more lenient threshold
    selector = BackwardFeatureSelector(variance_threshold=0.001)  # Reduced from 0.01
    
    # Analyze low activity pixels with more lenient thresholds
    pixels_to_remove, keep_mask = selector.analyze_low_activity_pixels(
        X_train, 
        intensity_threshold=0.02,  # Reduced from 0.05
        occurrence_threshold=0.01   # Reduced from 0.02
    )

    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Number of consistently low-intensity pixels: {len(pixels_to_remove)}")
    print(f"Pixels to remove (first 10): {pixels_to_remove[:10]}...")

    # Remove low activity pixels from both training and test sets
    X_train_filtered = np.delete(X_train, pixels_to_remove, axis=1)
    X_test_filtered = np.delete(X_test, pixels_to_remove, axis=1)

    # Keep more features after filtering
    remaining_features_to_keep = 600  # Increased from 457
    if remaining_features_to_keep > 0:
        X_train_selected, selected_features, n_selected = selector.select_features(
            X_train_filtered, y_train, n_features_to_keep=remaining_features_to_keep
        )
        # Apply the same feature selection to test set
        X_test_selected = X_test_filtered[:, selected_features]

        print(f"\nAfter feature selection:")
        print(f"Number of selected features: {n_selected}")
        print(f"Selected feature indices: {selected_features[:10]}...")
        
        # Save the processed data
        selector.save_processed_data(
            X_train_selected, X_test_selected, y_train, y_test, selected_features
        )

    # Calculate and print variance statistics
    print(f"\nVariance statistics of original features:")
    print(f"Mean variance: {np.mean(selector.feature_variances):.4f}")
    print(f"Max variance: {np.max(selector.feature_variances):.4f}")
    print(f"Min variance: {np.min(selector.feature_variances):.4f}")


if __name__ == "__main__":
    # Create processed_data directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)
    main()
