"""Unit tests for utility functions."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import (
    load_and_preprocess_data,
    preprocess_image,
    get_class_name
)
from config import CLASS_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH


class TestDataLoading:
    """Test data loading functions."""
    
    def test_load_and_preprocess_data(self):
        """Test data loading and preprocessing."""
        train_images, train_labels, test_images, test_labels = load_and_preprocess_data()
        
        # Check shapes
        assert train_images.ndim == 4
        assert test_images.ndim == 4
        assert train_labels.ndim == 1
        assert test_labels.ndim == 1
        
        # Check normalization
        assert train_images.max() <= 1.0
        assert train_images.min() >= 0.0
        assert test_images.max() <= 1.0
        assert test_images.min() >= 0.0
        
        # Check data types
        assert train_images.dtype == np.float32
        assert test_images.dtype == np.float32


class TestImagePreprocessing:
    """Test image preprocessing functions."""
    
    def test_preprocess_image_2d(self):
        """Test preprocessing 2D image."""
        image = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.float32)
        processed = preprocess_image(image)
        
        assert processed.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        assert processed.dtype == np.float32
    
    def test_preprocess_image_3d(self):
        """Test preprocessing 3D image."""
        image = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, 1).astype(np.float32)
        processed = preprocess_image(image)
        
        assert processed.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    
    def test_preprocess_image_normalization(self):
        """Test image normalization."""
        image = np.random.randint(0, 256, (IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.uint8)
        processed = preprocess_image(image)
        
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_class_name_valid(self):
        """Test getting valid class name."""
        for i in range(len(CLASS_NAMES)):
            name = get_class_name(i)
            assert name == CLASS_NAMES[i]
            assert isinstance(name, str)
    
    def test_get_class_name_invalid(self):
        """Test getting invalid class name."""
        with pytest.raises(IndexError):
            get_class_name(len(CLASS_NAMES))
        
        with pytest.raises(IndexError):
            get_class_name(-100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
