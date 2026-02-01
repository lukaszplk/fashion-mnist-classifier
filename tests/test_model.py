"""Unit tests for model architecture."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import create_simple_model, create_cnn_model, compile_model, get_model
from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES


class TestModelCreation:
    """Test model creation functions."""
    
    def test_create_simple_model(self):
        """Test simple model creation."""
        model = create_simple_model()
        assert model is not None
        assert len(model.layers) > 0
        assert model.name == "simple_dense_model"
    
    def test_create_cnn_model(self):
        """Test CNN model creation."""
        model = create_cnn_model()
        assert model is not None
        assert len(model.layers) > 0
        assert model.name == "fashion_mnist_cnn"
    
    def test_model_input_shape(self):
        """Test model input shape."""
        model = create_cnn_model()
        input_shape = model.input_shape
        assert input_shape[1:] == (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
    
    def test_model_output_shape(self):
        """Test model output shape."""
        model = create_cnn_model()
        output_shape = model.output_shape
        assert output_shape[-1] == NUM_CLASSES
    
    def test_compile_model(self):
        """Test model compilation."""
        model = create_cnn_model()
        compiled_model = compile_model(model)
        assert compiled_model.optimizer is not None
        assert compiled_model.loss is not None
        assert len(compiled_model.metrics) > 0
    
    def test_get_model_simple(self):
        """Test getting simple model."""
        model = get_model("simple")
        assert model is not None
        assert model.optimizer is not None
    
    def test_get_model_cnn(self):
        """Test getting CNN model."""
        model = get_model("cnn")
        assert model is not None
        assert model.optimizer is not None
    
    def test_get_model_invalid(self):
        """Test invalid model type."""
        with pytest.raises(ValueError):
            get_model("invalid_model_type")


class TestModelPrediction:
    """Test model prediction capabilities."""
    
    def test_model_prediction_shape(self):
        """Test prediction output shape."""
        model = get_model("cnn")
        dummy_input = np.random.rand(1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS).astype(np.float32)
        predictions = model.predict(dummy_input, verbose=0)
        assert predictions.shape == (1, NUM_CLASSES)
    
    def test_model_batch_prediction(self):
        """Test batch prediction."""
        model = get_model("cnn")
        batch_size = 5
        dummy_input = np.random.rand(batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS).astype(np.float32)
        predictions = model.predict(dummy_input, verbose=0)
        assert predictions.shape == (batch_size, NUM_CLASSES)
    
    def test_prediction_range(self):
        """Test that predictions are in valid range after softmax."""
        import tensorflow as tf
        model = get_model("cnn")
        dummy_input = np.random.rand(1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS).astype(np.float32)
        logits = model.predict(dummy_input, verbose=0)
        probabilities = tf.nn.softmax(logits).numpy()
        
        # Check probabilities sum to 1
        assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)
        # Check all probabilities are between 0 and 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
