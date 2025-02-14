# ANN-MLP
Artificial neural networks with a multilayer perceptron
# Python-Powered Handwritten Digit Recognition (with Flair!)

This project builds a Multi-Layer Perceptron (MLP) using TensorFlow and Keras in Python to recognize handwritten digits from the MNIST dataset.  It goes beyond the basics with opportunities for exploration and a creative extension.

## Project Objectives

*   Build a working MLP that achieves high accuracy on the MNIST dataset.
*   Understand the core concepts of neural networks (layers, activation functions, backpropagation, etc.) through practical implementation.
*   Learn how to use TensorFlow and Keras for building and training neural networks.
*   Explore different network architectures and hyperparameters.
*   Add a creative extension to the basic MNIST example.
*   Practice good coding habits and effective use of Cursor.

## Core Technologies

*   **Python:** The primary programming language.
*   **TensorFlow:** Google's open-source machine learning library.  We'll use the Keras API.
*   **Keras:** A high-level API for building neural networks, running on top of TensorFlow.
*   **Jupyter Notebook (via Cursor):** An interactive environment for running Python code, visualizing data, and documenting the process.
*   **MNIST Dataset:** The classic handwritten digit dataset.
*   **NumPy:** For numerical operations.
*   **Matplotlib:** For visualizing data and results.

## Learning Resources

*   **CS229 Lecture 2:** Provides the theoretical foundation for neural networks.
*   **TensorFlow and Keras Documentation:**  Essential for understanding the API.  Use `@TensorFlow Docs` and `@Keras Docs` in Cursor.
*   **TensorFlow Tutorials:** Start with the "Keras quickstart for beginners" and "Keras overview": [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)
*   **Deep Learning with Python (book by François Chollet):** Highly recommended for a deeper dive.
*    **CodeCademy:** Courses for refreshing Python.
*   **3Blue1Brown's Neural Networks Series (YouTube):**  For intuitive understanding.

## Project Steps & Vibe Coding Rules

This project follows the "15 Rules of Vibe Coding" and uses Cursor extensively.  Here's a breakdown:

1.  **Project Setup:**
    *   Create a new Jupyter Notebook: `mnist_mlp.ipynb`.
    *   Create a `.cursorrules` file (see below).
    *   Enable "Include .cursorrules file" in Cursor settings.
    *   Consider cloning a basic MNIST MLP Keras example from GitHub as a starting point (but don't be afraid to modify it!).

2.  **Imports and Data Loading:**
    *   New Composer chat.
    *   Prompt Cursor for necessary imports (`@codebase What libraries do I need to import...?`).
    *   Load MNIST data using `keras.datasets.mnist.load_data()` (prompt Cursor for help).
    *   Explore data shapes and visualize sample images (prompt Cursor).
    *   Run and inspect data frequently.
    *   Tag TensorFlow and Keras documentation in Cursor.

3.  **Data Preprocessing:**
    *   New Composer chat.
    *   Normalize pixel data (divide by 255.0 - ask Cursor why).
    *   Reshape images into vectors (ask Cursor).
    *   One-hot encode labels using `keras.utils.to_categorical` (ask Cursor).

4.  **Model Definition:**
    *   New Composer chat.
    *   Define the MLP architecture using `tf.keras.Sequential` (prompt Cursor, using Perplexity for ideas).  Start with one hidden layer. Use ReLU and softmax.
    *   *Carefully review* the generated code.

5.  **Model Compilation:**
    *   New Composer chat.
    *   Compile the model (`model.compile`) with appropriate optimizer ('adam'), loss ('categorical_crossentropy'), and metrics ('accuracy') – ask Cursor for explanations.
    *   Use `@TensorFlow Docs` and `@Keras Docs` to understand options.

6.  **Model Training:**
    *   New Composer chat.
    *   Train the model (`model.fit()`) with appropriate batch size and epochs (ask Cursor).
    *   Handle any errors by pasting them into the Composer chat.

7.  **Model Evaluation:**
    *   New Composer chat.
    *   Evaluate the model on test data (`model.evaluate()`).
    *   Visualize predictions and misclassified examples using Matplotlib (prompt Cursor).

8.  **Creative Extension:**  Choose *at least one*:
    *   Visualize weights.
    *   Generate adversarial examples.
    *   Experiment with different architectures (more layers, different activation functions).
    *   Implement a Convolutional Neural Network (CNN).
    *   Compare different optimizers (SGD, RMSprop).
    *   Add regularization (L1, L2, Dropout).

9.  **Saving and Loading (Optional):** Save and load your trained model.

**Key Prompts (Keep a record of these!):**

*   "What libraries do I need to import...?"
*   "How do I load the MNIST dataset using Keras?"
*   "How do I normalize pixel data in NumPy?"
*   "Explain one-hot encoding and how to do it in Keras."
*   "Create a Keras MLP with one hidden layer for MNIST. Use ReLU and softmax."
*   "How do I compile a Keras model? Suggest optimizer, loss, and metrics."
*   "How do I train a Keras model using `model.fit()`?"
*   "How do I evaluate a Keras model on test data?"
*   `@Keras Docs Explain the 'adam' optimizer.`
*   `@TensorFlow Docs What does the tf.keras.layers.Dense layer do?`

## `.cursorrules` File (Important!)

Create a file named `.cursorrules` in the same directory as your notebook.  Paste the following into it: