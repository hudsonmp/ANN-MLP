import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_digit(image, actual_label=None, predicted_label=None):
    """Plot a single MNIST digit.
    
    Args:
        image: 784-dimensional array representing the digit
        actual_label: True label of the digit (optional)
        predicted_label: Predicted label of the digit (optional)
    """
    # Reshape the image to 28x28
    img = image.reshape(28, 28)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    title = ''
    if actual_label is not None:
        title += f'Actual: {actual_label}'
    if predicted_label is not None:
        title += f' Predicted: {predicted_label}'
    
    plt.title(title)
    plt.show()

def plot_multiple_digits(images, labels=None, predictions=None, num_images=10):
    """Plot multiple MNIST digits in a grid.
    
    Args:
        images: Array of digit images
        labels: Array of true labels (optional)
        predictions: Array of predicted labels (optional)
        num_images: Number of images to plot
    """
    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(1, num_images, figsize=(2*num_images, 2))
    
    for i in range(num_images):
        ax = axes[i] if num_images > 1 else axes
        img = images[i].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        title = ''
        if labels is not None:
            title += f'A: {labels[i]}'
        if predictions is not None:
            title += f'\nP: {predictions[i]}'
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def plot_training_history(loss_history):
    """Plot the training loss history.
    
    Args:
        loss_history: List of loss values during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred)) 