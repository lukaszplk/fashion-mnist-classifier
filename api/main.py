"""FastAPI web service for Fashion MNIST classification."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import API_HOST, API_PORT, API_TITLE, API_VERSION, CLASS_NAMES, MODELS_DIR
from predict import FashionMNISTPredictor

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="API for Fashion MNIST image classification using deep learning"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: str
    predicted_index: int
    confidence: float
    all_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor
    
    # Find the latest model
    model_files = list(MODELS_DIR.glob("*.h5"))
    
    if not model_files:
        print("WARNING: No model found! Please train a model first.")
        return
    
    # Use the most recent model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading model: {latest_model}")
    predictor = FashionMNISTPredictor(str(latest_model))
    print("Model loaded successfully!")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fashion MNIST Classifier API",
        "version": API_VERSION,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "classes": "/classes"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "model not loaded",
        "model_loaded": predictor is not None
    }


@app.get("/classes", response_model=Dict[str, List[str]])
async def get_classes():
    """Get all available class names."""
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image.
    
    Args:
        file: Uploaded image file (should be grayscale, will be resized to 28x28)
        
    Returns:
        Prediction results with class name, index, and probabilities
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to grayscale and resize
        image = image.convert('L')
        image = image.resize((28, 28))
        
        # Convert to numpy array
        image_array = np.array(image).astype('float32') / 255.0
        
        # Make prediction
        pred_idx, pred_name, confidence, probabilities = predictor.predict(
            image_array,
            return_probabilities=True
        )
        
        # Format probabilities
        prob_dict = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            "predicted_class": pred_name,
            "predicted_index": int(pred_idx),
            "confidence": float(confidence),
            "all_probabilities": prob_dict
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict classes for multiple images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    results = []
    
    for file in files:
        try:
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Convert to grayscale and resize
            image = image.convert('L')
            image = image.resize((28, 28))
            
            # Convert to numpy array
            image_array = np.array(image).astype('float32') / 255.0
            
            # Make prediction
            pred_idx, pred_name, confidence, probabilities = predictor.predict(
                image_array,
                return_probabilities=True
            )
            
            # Format probabilities
            prob_dict = {
                CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(CLASS_NAMES))
            }
            
            results.append({
                "predicted_class": pred_name,
                "predicted_index": int(pred_idx),
                "confidence": float(confidence),
                "all_probabilities": prob_dict
            })
            
        except Exception as e:
            results.append({
                "predicted_class": "error",
                "predicted_index": -1,
                "confidence": 0.0,
                "all_probabilities": {},
                "error": str(e)
            })
    
    return results


if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting Fashion MNIST Classifier API on {API_HOST}:{API_PORT}")
    print(f"API Documentation: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)
