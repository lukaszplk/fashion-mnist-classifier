# 👕 Fashion MNIST Classifier

A production-ready deep learning project for classifying fashion items using the Fashion MNIST dataset. Features a modern CNN architecture, RESTful API, interactive web interface, and complete Docker support.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

- **Modern CNN Architecture**: Deep learning model with batch normalization, dropout, and advanced regularization
- **RESTful API**: FastAPI-powered API with automatic documentation
- **Interactive Web UI**: Gradio interface for easy image upload and predictions
- **Comprehensive Visualization**: Training history, confusion matrices, and sample predictions
- **Docker Support**: Containerized deployment for production environments
- **Unit Tests**: Full test coverage for model and utilities
- **Modular Code**: Clean separation of concerns with proper project structure

## 📊 Dataset

Fashion MNIST consists of 70,000 grayscale images (28x28 pixels) of 10 fashion categories:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## 📁 Project Structure

```
image_rec/
├── src/                      # Source code
│   ├── config.py            # Configuration and hyperparameters
│   ├── model.py             # Model architecture
│   ├── train.py             # Training script
│   ├── predict.py           # Prediction/inference
│   └── utils.py             # Utility functions
├── api/                      # Web services
│   ├── main.py              # FastAPI service
│   └── gradio_app.py        # Gradio web interface
├── tests/                    # Unit tests
│   ├── test_model.py
│   └── test_utils.py
├── docker/                   # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── notebooks/                # Jupyter notebooks
│   └── image_rec.ipynb
├── models/                   # Saved models (generated)
├── logs/                     # Training logs and visualizations
├── data/                     # Dataset cache
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
└── README.md                # This file
```

## 🚀 Quick Start

### Option 1: Local Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd image_rec
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
cd src
python train.py --model-type cnn --epochs 30
```

5. **Run the API**
```bash
cd api
python main.py
```

Access the API at: http://localhost:8000/docs

6. **Run the Web Interface**
```bash
cd api
python gradio_app.py
```

Access the web UI at: http://localhost:7860

### Option 2: Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose -f docker/docker-compose.yml up --build
```

This starts:
- FastAPI service at http://localhost:8000
- Gradio web interface at http://localhost:7860

2. **Run individual services**
```bash
# API only
docker-compose -f docker/docker-compose.yml up api

# Gradio UI only
docker-compose -f docker/docker-compose.yml up gradio
```

## 📚 Usage

### Training

**Basic training:**
```bash
python src/train.py
```

**Advanced options:**
```bash
python src/train.py \
    --model-type cnn \
    --epochs 50 \
    --batch-size 64 \
    --no-viz  # Disable visualizations
```

**Training output:**
- Model saved to `models/`
- Training history plots in `logs/visualizations/`
- TensorBoard logs in `logs/`
- Confusion matrix and sample predictions

### Prediction

**Predict on test dataset:**
```bash
python src/predict.py --model models/best_model_cnn_*.h5 --num-samples 10
```

**Predict on custom image:**
```bash
python src/predict.py --model models/best_model_cnn_*.h5 --image path/to/image.png
```

### API Usage

**Health check:**
```bash
curl http://localhost:8000/health
```

**Get available classes:**
```bash
curl http://localhost:8000/classes
```

**Predict (using curl):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

**Python example:**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("image.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Testing

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_model.py -v
```

**With coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
```

## 🏗️ Model Architecture

### CNN Model (Recommended)

```
Input (28x28x1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv2D(128) → BatchNorm → Dropout(0.3)
    ↓
Flatten → Dense(128) → BatchNorm → Dropout(0.3)
    ↓
Output (10 classes)
```

**Key features:**
- Multiple convolutional blocks with increasing filters
- Batch normalization for training stability
- Dropout for regularization
- Achieves ~92-93% test accuracy

## 📈 Performance

| Model | Test Accuracy | Parameters | Training Time (30 epochs) |
|-------|--------------|------------|---------------------------|
| Simple Dense | ~88% | ~100K | ~2 minutes |
| CNN (Default) | ~92-93% | ~500K | ~10 minutes |

*Tested on: CPU (Intel i7), GPU training significantly faster*

## 🔧 Configuration

Edit `src/config.py` to customize:

- Model architecture (filters, units, dropout rate)
- Training hyperparameters (batch size, epochs, learning rate)
- Callback settings (early stopping, learning rate reduction)
- API configuration (host, port)

## 📊 Visualization

Training produces several visualizations:

1. **Training History**: Loss and accuracy curves
2. **Confusion Matrix**: Model performance per class
3. **Sample Predictions**: Visual inspection of predictions
4. **TensorBoard Logs**: Interactive training monitoring

**View TensorBoard:**
```bash
tensorboard --logdir logs/
```

## 🐳 Docker Commands

**Build image:**
```bash
docker build -f docker/Dockerfile -t fashion-mnist:latest .
```

**Run API container:**
```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models fashion-mnist:latest
```

**Stop all containers:**
```bash
docker-compose -f docker/docker-compose.yml down
```

## 🧪 Development

**Install development dependencies:**
```bash
pip install -r requirements.txt
pip install jupyter ipykernel pytest black flake8
```

**Format code:**
```bash
black src/ tests/
```

**Lint code:**
```bash
flake8 src/ tests/
```

**Install package in editable mode:**
```bash
pip install -e .
```

## 📝 API Endpoints

### FastAPI (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/classes` | Get all class names |
| POST | `/predict` | Predict single image |
| POST | `/predict/batch` | Predict multiple images |

Interactive API docs: http://localhost:8000/docs

### Gradio UI (Port 7860)

- Upload images for classification
- Draw fashion items with sketch pad
- View prediction probabilities for all classes

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Fashion MNIST dataset by Zalando Research
- TensorFlow and Keras teams
- FastAPI and Gradio developers

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ using TensorFlow, FastAPI, and Gradio**
