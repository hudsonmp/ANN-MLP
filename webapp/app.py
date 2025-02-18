from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from sklearn.datasets import fetch_openml
import os
import sys
import random
import base64
from PIL import Image
import io

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
        
        print(f"Model loaded successfully with {len(selected_features)} features!")
        print(f"Model architecture: input_size={weights_data['input_size']}, hidden_size={weights_data['hidden_size']}")
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

# Convert numpy array to base64 image
def array_to_base64(array):
    # Reshape and scale to 0-255
    image = (array.reshape(28, 28) * 255).astype(np.uint8)
    # Convert to PIL Image
    img = Image.fromarray(image)
    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    # Convert to base64
    return base64.b64encode(buffer.getvalue()).decode()

# Load the model and test data
model = None
X_test = None
y_test = None
selected_features = None

def initialize():
    global model, X_test, y_test, selected_features
    model, selected_features = load_model_and_features()
    X_test, y_test = load_mnist_data()
    print(f"Initialization complete. Model will use {len(selected_features)} features for prediction.")

initialize()

# Get random samples for the game
@app.route('/get_samples', methods=['GET'])
def get_samples():
    try:
        # Get 9 random indices for a 3x3 grid
        sample_indices = random.sample(range(len(X_test)), 9)
        samples = []
        
        for idx in sample_indices:
            image = X_test[idx]
            label = int(y_test[idx])
            
            samples.append({
                'id': idx,
                'image': array_to_base64(image),
                'label': label
            })
        
        return jsonify(samples)
    except Exception as e:
        print(f"Error getting samples: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Make prediction for a sample
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        
        # Convert back to numpy array
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image.convert('L')) / 255.0
        image_array = image_array.reshape(-1)
        
        # Select features and make prediction
        image_selected = image_array[selected_features]
        pred_proba = model.predict_proba(image_selected.reshape(1, -1))
        predicted_label = int(np.argmax(pred_proba))
        confidence = float(pred_proba[0][predicted_label])
        
        return jsonify({
            'prediction': predicted_label,
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