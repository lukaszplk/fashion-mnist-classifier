"""Model architecture for Fashion MNIST classification."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple

from config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CHANNELS,
    NUM_CLASSES,
    CONV_FILTERS,
    CONV_KERNEL_SIZE,
    POOL_SIZE,
    DROPOUT_RATE,
    DENSE_UNITS,
    LEARNING_RATE
)


def create_simple_model() -> keras.Model:
    """
    Create a simple dense neural network model.
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Flatten(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES)
    ], name="simple_dense_model")
    
    return model


def create_cnn_model() -> keras.Model:
    """
    Create an improved CNN model with batch normalization and dropout.
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)),
        
        # First convolutional block
        layers.Conv2D(CONV_FILTERS[0], CONV_KERNEL_SIZE, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(CONV_FILTERS[0], CONV_KERNEL_SIZE, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(POOL_SIZE),
        layers.Dropout(DROPOUT_RATE),
        
        # Second convolutional block
        layers.Conv2D(CONV_FILTERS[1], CONV_KERNEL_SIZE, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(CONV_FILTERS[1], CONV_KERNEL_SIZE, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(POOL_SIZE),
        layers.Dropout(DROPOUT_RATE),
        
        # Third convolutional block
        layers.Conv2D(CONV_FILTERS[2], CONV_KERNEL_SIZE, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(DENSE_UNITS, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(NUM_CLASSES)
    ], name="fashion_mnist_cnn")
    
    return model


def compile_model(model: keras.Model, learning_rate: float = LEARNING_RATE) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )
    
    return model


def get_model(model_type: str = "cnn") -> keras.Model:
    """
    Get and compile a model.
    
    Args:
        model_type: Type of model - "simple" or "cnn"
        
    Returns:
        Compiled Keras model
    """
    if model_type == "simple":
        model = create_simple_model()
    elif model_type == "cnn":
        model = create_cnn_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'simple' or 'cnn'.")
    
    return compile_model(model)


if __name__ == "__main__":
    # Test model creation
    print("Creating Simple Model:")
    simple_model = get_model("simple")
    simple_model.summary()
    
    print("\n" + "="*70 + "\n")
    
    print("Creating CNN Model:")
    cnn_model = get_model("cnn")
    cnn_model.summary()
