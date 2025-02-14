Project Setup (Rules 1, 2, 9):

New Jupyter Notebook: Create a new file named mnist_mlp_from_scratch.ipynb.

.cursorrules (Rule 2):  Create a .cursorrules file. This is crucial for guiding Cursor, as we're doing something very non-standard.  Here's the adapted content:

You are a helpful coding assistant specialized in building neural networks from scratch using *only* Python and NumPy.
- **Do not use TensorFlow or Keras.**  Implement all neural network functionality manually.
- Use clear and concise variable names.
- Explain complex code sections thoroughly with comments. Include mathematical explanations where relevant.
- This project is for handwritten digit recognition using an MLP with the MNIST dataset.
- When creating a function, always include type hints.
- Use NumPy for all array manipulations.
- Use Matplotlib for visualizations.
- Assume the MNIST dataset is loaded using the `python-mnist` library.
- Provide implementations for:
    - Sigmoid, ReLU, and softmax activation functions (and their derivatives).
    - Forward propagation.
    - Backpropagation.
    - Gradient Descent (or a variant like mini-batch gradient descent).
    - A function to calculate accuracy.
- When defining the network architecture, explain the choices (number of layers, neurons).
- Explain the concepts of epochs, batch size, and learning rate.
- Be prepared to suggest and implement different MLP architectures.
- Be prepared to implement weight initialization strategies (e.g., Xavier/Glorot initialization).
- Guide the user through the mathematical derivations where appropriate.
- Generate clean, well-documented, and efficient NumPy code.
- For any prompt, write all code from scratch.

# Jupyter Notebook Specific:

- Generate code that is ready to be run directly in a Jupyter Notebook cell.
- Include Markdown cells for explanations, headings, and context where appropriate.
- Use `plt.show()` after Matplotlib plotting commands.

# Style and Readability
- Use consistent indentation (4 spaces).
- Limit line length to 79 characters where possible, following PEP 8 guidelines.
- Add blank lines between logical blocks of code.

# Error Handling
- If a user reports a runtime error, ask for the *full* error message.
- Try to identify the root cause before suggesting a solution.
- If a suggestion doesn't work, ask for details on what happened.

# Communication
- Be concise, but thorough. Explain the *why* behind the code.
- Use clear and unambiguous language.
- Maintain a professional and helpful tone.

Enable .cursorrules: Make sure it's enabled in Cursor's settings.

No Templates: Because this is a "from scratch" project, we won't start from a code template.

Imports and Data Loading (Rules 4, 6):

New Chat: Start a new Composer chat.
Imports: Ask Cursor: @codebase What libraries do I need to import for building an MLP from scratch with NumPy and Matplotlib, and loading MNIST withpython-mnist? Provide the import statements.
Load MNIST: Ask Cursor: @codebase How can I load the MNIST dataset using thepython-mnistlibrary? Provide code to load the training and testing data into NumPy arrays. This is different from the Keras method. You'll likely use the MNIST class from the library.
Data Exploration: Same as before – display shapes, visualize samples.
Data Preprocessing (Rules 3, 5):

New Chat: New chat for preprocessing.
Normalization: Ask Cursor: @codebase How should I normalize the MNIST pixel data? Provide the Python code using NumPy. (Still divide by 255.0).
Reshaping: Ask Cursor: @codebase Reshape the MNIST images into vectors for the MLP. Provide the NumPy code. (Same as before: (number of samples, 784)).
One-Hot Encoding: Ask Cursor: @codebase Explain one-hot encoding, and implement it in Python using *only* NumPy. This is now a manual implementation, not a Keras utility function.
Activation Functions (Rules 4, 5, 8):

New Chat: This is a major difference from the TensorFlow version.
Implement Functions: Ask Cursor: @codebase Implement the sigmoid, ReLU, and softmax activation functions, *and their derivatives*, in Python using NumPy. Include clear explanations of the mathematical formulas. This is critical for backpropagation. You'll need functions like sigmoid(x), sigmoid_derivative(x), relu(x), relu_derivative(x), softmax(x).
Review Carefully (Rules 7 & 8): Make absolutely sure the code for the activation functions and their derivatives is correct. Errors here will break the entire network.
Weight Initialization (Rules 4, 5):

New Chat:
Initialization: Ask Cursor: @codebase Explain the importance of proper weight initialization in neural networks. Implement Xavier/Glorot initialization for the weights of my MLP using NumPy. Assume I have a list of layer sizes (e.g.,[784, 128, 10]for an input layer, one hidden layer with 128 neurons, and an output layer). Random initialization is not sufficient; you need a strategy like Xavier/Glorot to prevent vanishing/exploding gradients.
Forward Propagation (Rules 4, 5):

New Chat:
Implement Forward Propagation: Ask Cursor: @codebase Implement the forward propagation algorithm for an MLP in Python using NumPy. The function should take the input data and a list of weights and biases, and return the output of each layer (activations). Use the activation functions I defined earlier. This is where you implement the core matrix multiplications and activation function applications.
Backpropagation (Rules 4, 5, 8):

New Chat: This is the most challenging part.
Implement Backpropagation: Ask Cursor: @codebase Implement the backpropagation algorithm for an MLP in Python using NumPy. The function should take the input data, target labels, network weights and biases, and activations from the forward pass, and return the gradients of the weights and biases. Explain the chain rule and how it's applied. This involves calculating the derivatives of the loss function with respect to the weights and biases, using the chain rule and the derivatives of the activation functions.
Review and Understand: Spend significant time understanding this code. This is the heart of the learning process.
Gradient Descent (Rules 4, 5):

New Chat:
Implement Gradient Descent: Ask Cursor: @codebase Implement mini-batch gradient descent in Python using NumPy. The function should take the weights, biases, gradients, learning rate, and batch size, and update the weights and biases. This is where you update the network's parameters based on the calculated gradients.
Model Training (Rules 4, 9):

New Chat:
Training Loop: Ask Cursor: @codebase Combine the forward propagation, backpropagation, and gradient descent functions into a training loop. Include options for setting the number of epochs, batch size, and learning rate. Calculate and print the loss and accuracy after each epoch. This brings all the pieces together.
Model Evaluation (Rules 4, 6):

New Chat:
Accuracy Function: Ask Cursor: @codebase Implement a function to calculate the accuracy of the network on a given dataset.
Evaluate on Test Data: Use your accuracy function to evaluate the trained model on the test data.
Visualize Predictions: Display some example images, predictions, and true labels (as before).
Creative Extension (Rules 15):

The same general ideas apply (visualizing weights, different architectures, etc.), but now you'll be implementing them from scratch using NumPy. This will be much more challenging than using TensorFlow.
Key Prompts:

"What libraries do I need to import for building an MLP from scratch with NumPy and Matplotlib?"
"How can I load the MNIST dataset using the python-mnist library?"
"Implement the sigmoid/ReLU/softmax activation function and its derivative in Python using NumPy."
"Explain Xavier/Glorot weight initialization and implement it in NumPy."
"Implement forward propagation for an MLP in Python using NumPy."
"Implement backpropagation for an MLP in Python using NumPy. Explain the chain rule."
"Implement mini-batch gradient descent in Python using NumPy."
"Combine forward propagation, backpropagation, and gradient descent into a training loop."
"Implement a function to calculate the accuracy of the network."