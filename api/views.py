from django.shortcuts import render
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import base64
from PIL import Image
import io
import os
from pathlib import Path
from mnist_slp.single_layer_perceptron.single_layer import SingleLayerForward

# Initialize the model and load trained weights
model_dir = Path(__file__).resolve().parent.parent / 'mnist_slp/saved_models'
model_path = model_dir / 'single_layer_perceptron.npz'

print(f"Looking for model at: {model_path}")
if not model_path.exists():
    raise RuntimeError("Trained model parameters not found. Please run the training script first.")

# Load model
print("Loading model parameters...")
model_params = np.load(str(model_path))
print(f"Model parameters loaded. Keys: {model_params.files}")
print(f"Weights shape: {model_params['weights'].shape}")
print(f"Biases shape: {model_params['biases'].shape}")

n_features = model_params['weights'].shape[1]
model = SingleLayerForward(n_features, 10)
model.load_parameters(str(model_path))

def preprocess_image(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocess the image to match MNIST training data characteristics.
    MNIST images are centered around 0 and scaled appropriately.
    """
    # Ensure proper scaling to [0, 1]
    image_array = image_array.astype('float32') / 255.0
    
    # Invert the colors (MNIST has white digits on black background)
    image_array = 1.0 - image_array
    
    # Apply batch normalization
    mean = np.mean(image_array)
    std = np.std(image_array) + 1e-8
    image_array = (image_array - mean) / std
    
    return image_array

@csrf_exempt
@require_http_methods(["POST"])
def predict_digit(request):
    try:
        # Parse the JSON data from the request
        data = json.loads(request.body)
        image_data = data.get('image')
        
        # Remove the data URL prefix if present
        if image_data.startswith('data:image/png;base64,'):
            image_data = image_data.replace('data:image/png;base64,', '')
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize
        image = image.convert('L')
        image = image.resize((28, 28))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Debug: Print image statistics
        print(f"Image shape: {image_array.shape}")
        print(f"Image min/max values before preprocessing: {image_array.min()}, {image_array.max()}")
        
        # Preprocess to match MNIST characteristics
        image_array = preprocess_image(image_array)
        
        # Debug: Print preprocessed image statistics
        print(f"Image min/max values after preprocessing: {image_array.min()}, {image_array.max()}")
        print(f"Image mean: {np.mean(image_array)}, std: {np.std(image_array)}")
        
        # Reshape for model input (784, 1)
        image_array = image_array.reshape(784, 1)
        
        # Make prediction
        predictions, _ = model.forward(image_array)
        
        # Apply softmax to get proper probabilities
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / np.sum(exp_preds)
        
        # Debug: Print prediction probabilities
        print(f"Raw predictions: {probabilities.ravel()}")
        
        # Get predicted digit and confidence
        predicted_digit = np.argmax(probabilities)
        probs = probabilities.ravel()
        confidence = float(probs[predicted_digit])
        
        print(f"Predicted digit: {predicted_digit}, Confidence: {confidence}")
        
        return JsonResponse({
            'digit': int(predicted_digit),
            'confidence': confidence,
            'probabilities': probs.tolist()
        })
        
    except Exception as e:
        import traceback
        print(f"Error in predict_digit: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'error': str(e)
        }, status=400)
