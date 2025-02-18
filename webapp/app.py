from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import io
import base64
import os
import sys

# Add the parent directory to Python path to import the MLP
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_mlp.mlp import MultiLayerPerceptron

app = Flask(__name__)

# Load the trained model
def load_model():
    try:
        weights_data = np.load('../model/weights.npy', allow_pickle=True).item()
        model = MultiLayerPerceptron(
            input_size=weights_data['input_size'],
            hidden_size=weights_data['hidden_size'],
            output_size=weights_data['output_size']
        )
        model.weights = weights_data['weights']
        model.biases = weights_data['biases']
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

model = load_model()

# Serve static files
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image']
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image).reshape(1, -1) / 255.0
        
        # Select the same features as used in training
        selected_features = np.load('../model/weights.npy', allow_pickle=True).item()['input_size']
        
        # Make prediction
        prediction = model.predict_proba(image_array)
        predicted_digit = np.argmax(prediction)
        confidence = float(prediction[0][predicted_digit])
        
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 