import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import sys


def load_data():
    """
    Load MNIST data from scikit-learn.
    Returns the dataset for exploration.
    """
    try:
        print("Loading MNIST from scikit-learn (this may take a few minutes)...")
        X, y = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
        print("Dataset loaded successfully!")
        return X, y
    except Exception as e:
        print(f"Error loading dataset: {str(e)}", file=sys.stderr)
        raise


def explore_data_shapes(X, y):
    """
    Print shape information about the dataset.
    """
    try:
        print("\nData Shapes:")
        print(f"X shape (samples, features): {X.shape}")
        print(f"y shape (samples,): {y.shape}")
        print(f"\nFeature information:")
        print(
            f"Each image is {int(np.sqrt(X.shape[1]))}x{int(np.sqrt(X.shape[1]))} pixels"
        )
        print(f"Pixel values range from {X.min()} to {X.max()}")
    except Exception as e:
        print(f"Error exploring data shapes: {str(e)}", file=sys.stderr)
        raise


def visualize_sample_images(X, y, num_samples=5):
    """
    Visualize sample images from the dataset.
    """
    try:
        print("Generating sample images visualization...")
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        fig.suptitle("MNIST Sample Images")

        for i in range(num_samples):
            img = X[i].reshape(28, 28)
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"Digit: {y[i]}")
            axes[i].axis("off")

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing sample images: {str(e)}", file=sys.stderr)
        raise


def visualize_digit_distribution(y):
    """
    Create a histogram showing the distribution of digits in the dataset.
    """
    try:
        print("Generating digit distribution visualization...")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(y.astype(int), bins=np.arange(11) - 0.5, rwidth=0.8)
        ax.set_title("Distribution of Digits in MNIST Dataset")
        ax.set_xlabel("Digit")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing digit distribution: {str(e)}", file=sys.stderr)
        raise


def visualize_pixel_intensities(X):
    """
    Visualize the distribution of pixel intensities in the dataset.
    """
    try:
        print("Generating pixel intensities visualization...")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(X.ravel(), bins=50, density=True)
        ax.set_title("Distribution of Pixel Intensities")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Density")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing pixel intensities: {str(e)}", file=sys.stderr)
        raise


def save_figure(fig, filename):
    """
    Save a figure to a file with error handling.
    """
    try:
        print(f"Saving {filename}...")
        fig.savefig(filename)
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {str(e)}", file=sys.stderr)
        raise


def main():
    try:
        # Load data
        X, y = load_data()

        # Explore shapes
        explore_data_shapes(X, y)

        # Create and save visualizations
        print("\nGenerating visualizations...")

        sample_fig = visualize_sample_images(X, y)
        save_figure(sample_fig, "mnist_samples.png")
        plt.close(sample_fig)

        dist_fig = visualize_digit_distribution(y)
        save_figure(dist_fig, "digit_distribution.png")
        plt.close(dist_fig)

        intensity_fig = visualize_pixel_intensities(X)
        save_figure(intensity_fig, "pixel_intensities.png")
        plt.close(intensity_fig)

        print("\nAll visualization files have been saved successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
