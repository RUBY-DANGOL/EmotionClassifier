import os
import json
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# ----- Configuration -----
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(SCRIPT_DIR, "densenetmodel_classify.h5"))
LABELS_PATH = os.environ.get("LABELS_PATH", os.path.join(SCRIPT_DIR, "labels.json"))
TOP_K = int(os.environ.get("TOP_K", "5"))

print(f"[DEBUG] Script directory: {SCRIPT_DIR}")
print(f"[DEBUG] Looking for model at: {MODEL_PATH}")
print(f"[DEBUG] Looking for labels at: {LABELS_PATH}")

# ----- Model Loading -----
def load_legacy_h5(path):
    if not os.path.exists(path):
        print(f"[WARN] Model file '{path}' not found.")
        return None
    
    print(f"[INFO] Attempting to load model from: {path}")
    
    # Try different loading strategies
    strategies = [
        {"safe_mode": False, "compile": False, "name": "safe_mode=False, compile=False"},
        {"safe_mode": True, "compile": False, "name": "safe_mode=True, compile=False"},
        {"compile": False, "name": "compile=False only"},
        {"compile": True, "name": "compile=True"},
    ]
    
    for strategy in strategies:
        try:
            print(f"[INFO] Trying strategy: {strategy['name']}")
            m = tf.keras.models.load_model(path, **{k: v for k, v in strategy.items() if k != 'name'})
            print(f"[INFO] Successfully loaded model with strategy: {strategy['name']}")
            print(f"[INFO] Model input shape: {m.input_shape}")
            print(f"[INFO] Model output shape: {m.output_shape}")
            return m
        except Exception as e:
            print(f"[WARN] Strategy '{strategy['name']}' failed: {e}")
            continue
    
    # If all strategies fail, try loading just the weights
    try:
        print("[INFO] All standard loading methods failed. Attempting to create model architecture...")
        # Try to reconstruct the model based on the DenseNet architecture we see in the notebook
        from tensorflow.keras.applications.densenet import DenseNet121
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
        from tensorflow.keras.models import Sequential
        
        print("[INFO] Creating DenseNet121 base model...")
        base_model = DenseNet121(
            weights=None,  # Don't load pretrained weights yet
            include_top=False,
            input_shape=(256, 256, 3)
        )
        
        # Create the full model architecture
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(7, activation='softmax')  # 7 emotions
        ])
        
        print("[INFO] Loading weights into reconstructed model...")
        # Try to load just the weights
        model.load_weights(path)
        print("[INFO] Successfully loaded weights into reconstructed model!")
        return model
        
    except Exception as e2:
        print(f"[ERROR] Weight loading also failed: {e2}")
        print("[ERROR] Could not load model with any method.")
        return None

model = load_legacy_h5(MODEL_PATH)

# Check if model loaded successfully
def infer_input_shape():
    if model is None:
        return (224, 224, 3)
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) == 4:
        _, H, W, C = shape
        return (int(H), int(W), int(C))
    return (224, 224, 3)

# Always define these, even if model failed to load
try:
    INPUT_H, INPUT_W, INPUT_C = infer_input_shape()
except Exception:
    INPUT_H, INPUT_W, INPUT_C = (224, 224, 3)

# ----- Label Loading -----
def load_labels(path):
    if not os.path.exists(path):
        print(f"[WARN] Labels file '{path}' not found.")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"[INFO] Loaded {len(labels)} labels from {path}")
        return labels
    except Exception as e:
        print(f"[ERROR] Failed to load labels: {e}")
        return {}

labels = load_labels(LABELS_PATH)

# ----- Image Preprocessing -----
def preprocess_image(image, target_size):
    """
    Preprocess image for model prediction
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None

# ----- Prediction Function -----
def predict_image(image):
    """
    Make prediction on preprocessed image
    """
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Preprocess image
        processed_img = preprocess_image(image, (INPUT_W, INPUT_H))
        if processed_img is None:
            return {"error": "Image preprocessing failed"}
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Get top predictions
        if len(predictions.shape) > 1:
            pred_scores = predictions[0]
        else:
            pred_scores = predictions
        
        # Get top K predictions
        top_indices = np.argsort(pred_scores)[-TOP_K:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            score = float(pred_scores[idx])
            
            # Get label name
            if labels:
                label_keys = list(labels.keys())
                if idx < len(label_keys):
                    label_key = label_keys[idx]
                    label_name = labels[label_key]
                else:
                    label_name = f"Class_{idx}"
            else:
                label_name = f"Class_{idx}"
            
            results.append({
                "label": label_name,
                "confidence": score,
                "rank": i + 1
            })
        
        return {"predictions": results}
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

# ----- Flask Routes -----
@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html',
                         input_w=INPUT_W,
                         input_h=INPUT_H,
                         input_c=INPUT_C,
                         has_model=model is not None,
                         has_labels=bool(labels))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image prediction requests
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image/'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Make prediction
        result = predict_image(image)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload predictions
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Open and process image
        image = Image.open(file.stream)
        
        # Make prediction
        result = predict_image(image)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/health')
def health():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_loaded": bool(labels),
        "input_shape": f"{INPUT_W}x{INPUT_H}x{INPUT_C}"
    })

if __name__ == '__main__':
    print(f"[INFO] Starting Flask app...")
    print(f"[INFO] Model loaded: {model is not None}")
    print(f"[INFO] Labels loaded: {len(labels)} classes")
    print(f"[INFO] Input shape: {INPUT_W}x{INPUT_H}x{INPUT_C}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
