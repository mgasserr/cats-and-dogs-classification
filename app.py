"""
Cat vs Dog Image Classifier - Flask Web Application (PyTorch Version)
====================================================================
This application loads a pre-trained VGG16-based PyTorch model (saved from the
notebook) and provides a web interface where users can upload images to
classify them as Cat or Dog with confidence percentages.
"""

import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# ============================================
# PyTorch & Torchvision imports
# ============================================
import torch
import torchvision.transforms as transforms
from PIL import Image

# ============================================
# App Configuration
# ============================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# ============================================
# Model paths (relative to this file)
# ============================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cat_dog_vgg16_pytorch.pth')
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'training_history.json')

# ============================================
# Class names — must match the dataset order (0 = cats, 1 = dogs)
# ============================================
CLASS_NAMES = ['cats', 'dogs']

# ============================================
# Global variables (loaded once at startup)
# ============================================
model = None
training_history = None

# Define Image Preprocessing Transforms for VGG16
# These must match the normalizations used in PyTorch VGG16 training
vgg_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet standards
        std=[0.229, 0.224, 0.225]
    )
])


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_trained_model():
    """Load the saved PyTorch model from disk."""
    global model
    if not os.path.exists(MODEL_PATH):
        print(
            f"[WARN] Model file not found at: {MODEL_PATH}\n"
            "       Please run the PyTorch notebook first to train and save the model.\n"
            "       The web app will start, but predictions won't work until the model is available."
        )
        return False
    try:
        print(f"[INFO] Loading PyTorch model from {MODEL_PATH} ...")
        # Load to CPU to ensure it runs even if host machine does not use GPU for inference
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        model.eval()  # Set model to evaluation mode
        print("[INFO] PyTorch model loaded successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        return False


def load_training_history():
    """Load training history from JSON file."""
    global training_history
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            training_history = json.load(f)
        print("[INFO] Training history loaded successfully.")
    else:
        training_history = None
        print("[WARN] Training history file not found. Some stats won't be available.")


def predict_image(img_path):
    """
    Run the PyTorch model on a single image and return prediction results.
    """
    # Load image using PIL
    img = Image.open(img_path).convert('RGB')
    
    # Preprocess image
    img_tensor = vgg_transforms(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)

    # Perform Inference
    with torch.no_grad():
        output = model(img_tensor)
        # Assuming model outputs sigmoid probabilities (values between 0 and 1)
        pred_prob = float(output.item())

    # Sigmoid output: > 0.5 is Dog, <= 0.5 is Cat
    if pred_prob > 0.5:
        predicted_class = 'dogs'
        confidence = pred_prob * 100
    else:
        predicted_class = 'cats'
        confidence = (1 - pred_prob) * 100

    return {
        'class': predicted_class,
        'confidence': round(confidence, 2),
        'cat_probability': round((1 - pred_prob) * 100, 2),
        'dog_probability': round(pred_prob * 100, 2),
        'raw_prediction': round(pred_prob, 6),
    }


# ============================================
# Routes
# ============================================

@app.route('/')
def index():
    """Render the main upload page."""
    stats = None
    if training_history:
        stats = {
            'epochs': len(training_history.get('accuracy', [])),
            'final_train_acc': round(training_history['accuracy'][-1] * 100, 2) if training_history.get('accuracy') else None,
            'final_val_acc': round(training_history['val_accuracy'][-1] * 100, 2) if training_history.get('val_accuracy') else None,
            'final_train_loss': round(training_history['loss'][-1], 4) if training_history.get('loss') else None,
            'final_val_loss': round(training_history['val_loss'][-1], 4) if training_history.get('val_loss') else None,
        }
    model_loaded = model is not None
    return render_template('index.html', stats=stats, model_loaded=model_loaded, training_history=training_history)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please run the PyTorch notebook first to train and save the model, '
                     'then restart the Flask server.'
        }), 503

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = predict_image(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Commented out cleanup so you can see images saved in the uploads/ folder
        # if os.path.exists(filepath):
        #     os.remove(filepath)
        pass


@app.route('/stats')
def stats():
    """Return training statistics as JSON."""
    if training_history:
        return jsonify(training_history)
    return jsonify({'error': 'No training history available.'}), 404


# ============================================
# Main
# ============================================
if __name__ == '__main__':
    load_trained_model()
    load_training_history()
    print("[INFO] Starting Flask server ...")
    app.run(debug=True, host='0.0.0.0', port=5000)
