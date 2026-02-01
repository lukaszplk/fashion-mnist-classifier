"""Configuration file for the Fashion MNIST classifier."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "fashion_mnist_cnn"
MODEL_VERSION = "v1"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}_{MODEL_VERSION}.h5"

# Dataset configuration
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10

# Class names for Fashion MNIST
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Model architecture
CONV_FILTERS = [32, 64, 128]
CONV_KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DROPOUT_RATE = 0.3
DENSE_UNITS = 128

# Callbacks
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Fashion MNIST Classifier API"
API_VERSION = "1.0.0"
