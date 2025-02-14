import numpy as np
import pytest
from src.models.mlp import MLP
from src.utils.activations import relu, softmax

def test_mlp_initialization():
    layer_sizes = [2, 3, 1]
    model = MLP(layer_sizes)
    
    assert len(model.layers) == 2
    assert model.layers[0].weights.shape == (2, 3)
    assert model.layers[1].weights.shape == (3, 1)

def test_mlp_forward():
    layer_sizes = [2, 3, 2]
    model = MLP(layer_sizes)
    
    # Test with batch size of 1
    x = np.array([[1.0, 2.0]])
    output = model.forward(x)
    assert output.shape == (1, 2)
    
    # Test with batch size of 4
    x = np.random.randn(4, 2)
    output = model.forward(x)
    assert output.shape == (4, 2)

def test_mlp_predict():
    layer_sizes = [2, 3, 2]
    model = MLP(layer_sizes)
    
    x = np.random.randn(4, 2)
    predictions = model.predict(x)
    
    assert predictions.shape == (4,)
    assert np.all(predictions >= 0) and np.all(predictions < 2)
