# ANN-MLP
Artificial Neural Networks with a Multilayer Perceptron
# Python-Powered Handwritten Digit Recognition

This project builds a Multi-Layer Perceptron (MLP) from scratch using NumPy to recognize handwritten digits from the MNIST dataset. It focuses on understanding the core concepts through direct implementation.

## Project Objectives

*   Build a working MLP that achieves good accuracy on the MNIST dataset
*   Understand the core concepts of neural networks (layers, activation functions, backpropagation, etc.) through low-level implementation
*   Learn how to implement neural networks using NumPy and basic Python
*   Explore different network architectures and hyperparameters
*   Add creative extensions to the basic MNIST example
*   Practice good coding habits and software engineering principles

## Core Technologies

*   **Python:** The primary programming language
*   **NumPy:** For efficient numerical operations and matrix math
*   **scikit-learn:** For loading the MNIST dataset and preprocessing
*   **Matplotlib:** For visualizing data and results
*   **Jupyter Notebook:** For interactive development and experimentation

## Learning Resources

*   **CS229 Lecture 2:** Provides the theoretical foundation for neural networks
*   **3Blue1Brown's Neural Networks Series (YouTube):** For intuitive understanding of backpropagation
*   **Deep Learning from Scratch (book by Seth Weidman):** Excellent resource for implementing neural networks with NumPy
*   **NumPy Documentation:** Essential for understanding array operations
*   **CodeCademy:** Courses for refreshing Python fundamentals

## Project Steps

1.  **Project Setup:**
    *   Create project structure with src/, tests/, and notebooks/ directories
    *   Set up a virtual environment
    *   Install required packages (numpy, sklearn, matplotlib)
    *   Create initial test files

2.  **Data Loading:**
    *   Use scikit-learn's fetch_openml to load MNIST
    *   Explore data shapes and visualize sample images
    *   Split into training and test sets
    *   Run and inspect data frequently

3.  **Data Preprocessing:**
    *   Normalize pixel values (scale to [0,1])
    *   Reshape images into feature vectors
    *   Convert labels to appropriate format
    *   Consider dimensionality reduction techniques

4.  **MLP Implementation:**
    *   Implement the Layer class with weights and biases
    *   Implement forward propagation with activation functions
    *   Implement backward propagation
    *   Create the main MLP class to combine layers
    *   Write comprehensive unit tests

5.  **Training Loop:**
    *   Implement mini-batch gradient descent
    *   Add learning rate scheduling
    *   Track and plot loss during training
    *   Save checkpoints of model parameters

6.  **Model Evaluation:**
    *   Calculate accuracy on test set
    *   Implement confusion matrix visualization
    *   Analyze misclassified examples
    *   Compare performance with different architectures

7.  **Creative Extensions:** Choose at least one:
    *   Visualize learned weights
    *   Implement different optimization algorithms (SGD, RMSprop, Adam)
    *   Add regularization techniques (L1, L2, Dropout)
    *   Try different activation functions
    *   Experiment with batch normalization
    *   Add momentum to gradient descent

8.  **Documentation and Testing:**
    *   Write clear docstrings and comments
    *   Create comprehensive unit tests
    *   Document hyperparameter choices
    *   Profile code performance

**Key Implementation Steps:**

*   Implement Layer class with NumPy arrays
*   Implement activation functions (ReLU, softmax)
*   Implement forward and backward propagation
*   Create training loop with mini-batch gradient descent
*   Add proper error handling and logging
*   Optimize performance with vectorized operations
*   Write thorough tests for each component

## Project Structure

Create the following directory structure: