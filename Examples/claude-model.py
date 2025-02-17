# model.py
import numpy as np

class MLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # He initialization for better gradient flow
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.biases1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.biases2 = np.zeros((1, output_size))
        
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Store intermediate values for backpropagation
        self.layer1_input = np.dot(X, self.weights1) + self.biases1
        self.layer1_output = self.relu(self.layer1_input)
        self.layer2_input = np.dot(self.layer1_output, self.weights2) + self.biases2
        self.output = self.softmax(self.layer2_input)
        return self.output
    
    def backward(self, X, y, learning_rate=0.01):
        batch_size = X.shape[0]
        
        # Convert labels to one-hot encoding
        one_hot_y = np.zeros((batch_size, 10))
        one_hot_y[np.arange(batch_size), y] = 1
        
        # Gradients for output layer
        d_output = self.output - one_hot_y
        d_weights2 = np.dot(self.layer1_output.T, d_output) / batch_size
        d_biases2 = np.sum(d_output, axis=0, keepdims=True) / batch_size
        
        # Gradients for hidden layer
        d_hidden = np.dot(d_output, self.weights2.T) * self.relu_derivative(self.layer1_input)
        d_weights1 = np.dot(X.T, d_hidden) / batch_size
        d_biases1 = np.sum(d_hidden, axis=0, keepdims=True) / batch_size
        
        # Update weights and biases
        self.weights2 -= learning_rate * d_weights2
        self.biases2 -= learning_rate * d_biases2
        self.weights1 -= learning_rate * d_weights1
        self.biases1 -= learning_rate * d_biases1
    
    def save_weights(self, filepath):
        weights = {
            'w1': self.weights1,
            'b1': self.biases1,
            'w2': self.weights2,
            'b2': self.biases2
        }
        np.save(filepath, weights)
    
    def load_weights(self, filepath):
        weights = np.load(filepath, allow_pickle=True).item()
        self.weights1 = weights['w1']
        self.biases1 = weights['b1']
        self.weights2 = weights['w2']
        self.biases2 = weights['b2']