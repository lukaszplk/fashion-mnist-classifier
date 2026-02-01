# Fashion MNIST Classifier - Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Training

**Train the model (takes ~10 minutes):**
```bash
cd src
python train.py
```

This will:
- Download Fashion MNIST dataset
- Train a CNN model
- Save the best model to `models/`
- Generate visualizations in `logs/visualizations/`

## Prediction

**Test predictions on random samples:**
```bash
python src/predict.py --model models/best_model_cnn_*.h5 --num-samples 5
```

## Web Interface

**Start the Gradio web interface:**
```bash
python api/gradio_app.py
```

Then open: http://localhost:7860

- Upload or draw fashion items
- See predictions instantly

## API Service

**Start the FastAPI service:**
```bash
python api/main.py
```

API docs: http://localhost:8000/docs

**Test with curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_image.png"
```

## Docker

**Run everything with Docker:**
```bash
docker-compose -f docker/docker-compose.yml up
```

- API: http://localhost:8000
- Gradio UI: http://localhost:7860

## Testing

**Run tests:**
```bash
pytest tests/ -v
```

## Project Structure

```
image_rec/
├── src/           # Python modules
├── api/           # Web services
├── notebooks/     # Jupyter notebooks
├── tests/         # Unit tests
├── models/        # Saved models
└── docker/        # Docker files
```

## Need Help?

See the full README.md for detailed documentation.
