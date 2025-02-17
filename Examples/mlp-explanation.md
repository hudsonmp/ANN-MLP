# Understanding MLP Implementation from Scratch

## 1. Initialization Phase (Once at start)

```python
class MLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # He initialization for better gradient flow
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.biases1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.biases2 = np.zeros((1, output_size))
```

- **Weight Initialization**: Using He initialization (multiply by sqrt(2/n))
  - Helps prevent vanishing/exploding gradients
  - Better for ReLU activation functions
  - Done once when creating the model

- **Bias Initialization**: Start with zeros
  - One bias per neuron in each layer
  - Shape matches the number of neurons in the layer

## 2. Forward Propagation Phase (Every batch)

```python
def forward(self, X):
    # First layer
    self.layer1_input = np.dot(X, self.weights1) + self.biases1  # (batch_size, hidden_size)
    self.layer1_output = self.relu(self.layer1_input)            # ReLU activation
    
    # Output layer
    self.layer2_input = np.dot(self.layer1_output, self.weights2) + self.biases2
    self.output = self.softmax(self.layer2_input)                # Softmax for probabilities
    
    return self.output

def relu(self, x):
    return np.maximum(0, x)

def softmax(self, x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

Steps (repeated for each batch):
1. Input layer → Hidden layer:
   - Matrix multiplication: (batch_size, 784) × (784, 128) = (batch_size, 128)
   - Add biases
   - Apply ReLU activation

2. Hidden layer → Output layer:
   - Matrix multiplication: (batch_size, 128) × (128, 10) = (batch_size, 10)
   - Add biases
   - Apply softmax activation

## 3. Loss Calculation Phase (Every batch)

```python
def calculate_loss(self, y_true, y_pred):
    # Convert labels to one-hot encoding
    batch_size = y_true.shape[0]
    y_one_hot = np.zeros((batch_size, 10))
    y_one_hot[np.arange(batch_size), y_true] = 1
    
    # Cross-entropy loss
    loss = -np.sum(y_one_hot * np.log(y_pred + 1e-7)) / batch_size
    return loss
```

- Cross-entropy loss for classification
- Add small epsilon (1e-7) to prevent log(0)
- Average loss over the batch
- Done after each forward pass

## 4. Backward Propagation Phase (Every batch)

```python
def backward(self, X, y, learning_rate=0.01):
    batch_size = X.shape[0]
    
    # Convert labels to one-hot
    y_one_hot = np.zeros((batch_size, 10))
    y_one_hot[np.arange(batch_size), y] = 1
    
    # Output layer gradients
    d_output = self.output - y_one_hot                           # (batch_size, 10)
    d_weights2 = np.dot(self.layer1_output.T, d_output)         # (128, 10)
    d_biases2 = np.sum(d_output, axis=0, keepdims=True)        # (1, 10)
    
    # Hidden layer gradients
    d_hidden = np.dot(d_output, self.weights2.T)                # (batch_size, 128)
    d_hidden_relu = d_hidden * (self.layer1_output > 0)         # ReLU derivative
    d_weights1 = np.dot(X.T, d_hidden_relu)                     # (784, 128)
    d_biases1 = np.sum(d_hidden_relu, axis=0, keepdims=True)   # (1, 128)
    
    # Update weights and biases
    self.weights2 -= learning_rate * d_weights2
    self.biases2 -= learning_rate * d_biases2
    self.weights1 -= learning_rate * d_weights1
    self.biases1 -= learning_rate * d_biases1
```

Chain rule application (backward order):
1. Output layer:
   - Calculate error at output (predictions - true_labels)
   - Compute gradients for weights2 and biases2

2. Hidden layer:
   - Propagate error backward through weights2
   - Apply ReLU derivative
   - Compute gradients for weights1 and biases1

## 5. Training Loop (Multiple epochs)

```python
def train(self, X_train, y_train, epochs=10, batch_size=32, learning_rate=0.01):
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass
            output = self.forward(X_batch)
            
            # Calculate loss
            loss = self.calculate_loss(y_batch, output)
            total_loss += loss
            
            # Backward pass
            self.backward(X_batch, y_batch, learning_rate)
        
        # Print epoch statistics
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
```

Training process:
1. Repeat for each epoch:
   - Shuffle training data
   - Split into batches
   - For each batch:
     - Forward propagation
     - Calculate loss
     - Backward propagation
     - Update weights
   - Calculate and display epoch statistics

## Key Concepts and Tips

1. **Matrix Shapes**:
   - Input: (batch_size, 784) for MNIST
   - Hidden layer: (batch_size, 128)
   - Output: (batch_size, 10) for 10 digits
   - Keep track of shapes for debugging

2. **Numerical Stability**:
   - Subtract max in softmax
   - Add epsilon in log
   - Use He initialization

3. **Performance Optimization**:
   - Use vectorized operations (avoid loops)
   - Process in batches
   - Normalize input data

4. **Hyperparameters**:
   - Learning rate: typically 0.01-0.001
   - Batch size: typically 32-128
   - Number of epochs: monitor validation loss
   - Hidden layer size: experiment for best results

Would you like me to explain any of these phases in more detail, or shall we start implementing them step by step?