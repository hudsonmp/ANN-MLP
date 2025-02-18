from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from sklearn.datasets import fetch_openml
import os
import sys
import random

# Add the parent directory to Python path to import the MLP
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from mnist_mlp.mlp import MultiLayerPerceptron

app = Flask(__name__)

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Load the trained model and get selected features
def load_model_and_features():
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'weights.npy')
        features_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed_data', 'selected_features.npy')
        
        # Load model weights and architecture
        weights_data = np.load(model_path, allow_pickle=True).item()
        model = MultiLayerPerceptron(
            input_size=weights_data['input_size'],
            hidden_size=weights_data['hidden_size'],
            output_size=weights_data['output_size'],
            learning_rate=0.001,
            batch_size=32,
            max_iterations=1000
        )
        model.weights = weights_data['weights']
        model.biases = weights_data['biases']
        
        # Load selected features
        selected_features = np.load(features_path, allow_pickle=True)
        
        print("Model and features loaded successfully!")
        return model, selected_features
    except Exception as e:
        print(f"Error loading model and features: {str(e)}")
        raise

# Load MNIST data
def load_mnist_data():
    try:
        print("Loading MNIST dataset...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = X / 255.0  # Normalize pixel values
        y = y.astype(np.int32)
        
        # Use last 10,000 samples as test set
        X_test = X[-10000:]
        y_test = y[-10000:]
        
        print(f"Loaded {len(X_test)} test images")
        return X_test, y_test
    except Exception as e:
        print(f"Error loading MNIST data: {str(e)}")
        raise

# Load the model and test data
model = None
X_test = None
y_test = None
selected_features = None

def initialize():
    global model, X_test, y_test, selected_features
    model, selected_features = load_model_and_features()
    X_test, y_test = load_mnist_data()

initialize()

# Get random test samples (without predictions)
@app.route('/get_test_samples', methods=['GET'])
def get_test_samples():
    try:
        # Get 5 random indices
        sample_indices = random.sample(range(len(X_test)), 5)
        samples = []
        
        for idx in sample_indices:
            # Get the image and its true label
            image = X_test[idx]
            true_label = int(y_test[idx])
            
            samples.append({
                'id': idx,  # Include the index for reference
                'pixels': image.tolist(),  # Send full image for display
                'true_label': true_label
            })
        
        return jsonify(samples)
    except Exception as e:
        print(f"Error getting test samples: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Make prediction for a single sample
@app.route('/predict/<int:sample_id>', methods=['GET'])
def predict_sample(sample_id):
    try:
        # Get the full image
        image = X_test[sample_id]
        true_label = int(y_test[sample_id])
        
        # Select features for prediction
        image_selected = image[selected_features]
        
        # Make prediction
        pred_proba = model.predict_proba(image_selected.reshape(1, -1))
        predicted_label = int(np.argmax(pred_proba))
        confidence = float(pred_proba[0][predicted_label])
        
        return jsonify({
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Serve static files
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True) 