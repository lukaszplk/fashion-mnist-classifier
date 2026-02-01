"""Gradio web interface for Fashion MNIST classification."""

import gradio as gr
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import CLASS_NAMES, MODELS_DIR
from predict import FashionMNISTPredictor

# Load model
model_files = list(MODELS_DIR.glob("*.h5"))
if model_files:
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    predictor = FashionMNISTPredictor(str(latest_model))
    print(f"Model loaded: {latest_model}")
else:
    print("WARNING: No model found! Please train a model first.")
    predictor = None


def classify_image(image):
    """
    Classify an uploaded or drawn image.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Dictionary of class names and their probabilities
    """
    if predictor is None:
        # Return uniform probabilities to indicate no model
        return {name: 0.0 for name in CLASS_NAMES}
    
    try:
        # Convert to grayscale if needed
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image).convert('L')
            else:
                image = Image.fromarray(image)
        else:
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array
        image_array = np.array(image).astype('float32') / 255.0
        
        # Get prediction with probabilities
        _, _, _, probabilities = predictor.predict(
            image_array,
            return_probabilities=True
        )
        
        # Create probability dictionary
        result = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
        
        return result
        
    except Exception as e:
        # Return uniform probabilities on error
        print(f"Error during prediction: {str(e)}")
        return {name: 0.0 for name in CLASS_NAMES}


# Create Gradio interface
with gr.Blocks(title="Fashion MNIST Classifier", theme=gr.themes.Soft()) as demo:
    # Check if model is loaded
    model_status = "✅ Model loaded and ready!" if predictor is not None else "⚠️ No model found - please train a model first!"
    
    gr.Markdown(
        f"""
        # 👕 Fashion MNIST Classifier
        
        **Status:** {model_status}
        
        Upload or draw a fashion item image to classify it into one of 10 categories!
        
        **Categories:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
        
        {"**Note:** Train a model first by running: `python src/train.py`" if predictor is None else ""}
        """
    )
    
    with gr.Tab("Upload Image"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=300
                )
                upload_btn = gr.Button("Classify", variant="primary", size="lg")
            
            with gr.Column():
                output_label = gr.Label(
                    label="Prediction",
                    num_top_classes=10
                )
        
        upload_btn.click(
            fn=classify_image,
            inputs=input_image,
            outputs=output_label
        )
    
    with gr.Tab("Draw Image"):
        with gr.Row():
            with gr.Column():
                input_sketch = gr.Sketchpad(
                    type="pil",
                    label="Draw Fashion Item",
                    height=300,
                    width=300,
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=15)
                )
                draw_btn = gr.Button("Classify", variant="primary", size="lg")
            
            with gr.Column():
                output_sketch = gr.Label(
                    label="Prediction",
                    num_top_classes=10
                )
        
        draw_btn.click(
            fn=classify_image,
            inputs=input_sketch,
            outputs=output_sketch
        )
    
    with gr.Tab("Examples"):
        gr.Markdown(
            """
            ### How to use:
            
            1. **Upload Tab**: Upload a grayscale image of a fashion item (preferably 28x28 pixels)
            2. **Draw Tab**: Draw a fashion item using the sketch pad
            3. Click **Classify** to get predictions
            
            The model will show confidence scores for all 10 classes, sorted by probability.
            
            ### Tips for best results:
            - Use simple, clear images
            - Center the item in the frame
            - Use high contrast (white item on black background or vice versa)
            - The model works best with Fashion MNIST-style images
            """
        )
    
    gr.Markdown(
        """
        ---
        **Note:** This is a deep learning model trained on the Fashion MNIST dataset.
        Model architecture: CNN with batch normalization and dropout layers.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
