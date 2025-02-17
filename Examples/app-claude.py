# app.py
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import io
import base64
import os
from mnist_mlp.mlp import MultiLayerPerceptron as MLP

app = Flask(__name__)
model = MLP()
model.load_weights('model/weights.npy')

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
        image_array = np.array(image).reshape(1, 784) / 255.0
        # Make prediction
        prediction = model.forward(image_array)
        # Get the predicted digit and confidence
        predicted_digit = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_digit])
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)