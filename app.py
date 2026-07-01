import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"[GPU] Found {len(physical_devices)} GPU(s)")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Enabled: {gpu}")
else:
    print("[CPU] No GPU found, using CPU")

# Patch Keras layers to handle quantization_config
import keras
_original_dense_init = keras.layers.Dense.__init__
def _patched_dense_init(self, *args, quantization_config=None, **kwargs):
    _original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = _patched_dense_init

app = Flask(__name__)

# ==============================
# LOAD MODEL + LABELS
# ==============================
MODEL_PATH = "signbert_50_classes.keras"
LABEL_PATH = "label_encoder_50.pkl"

model = load_model(MODEL_PATH, safe_mode=False)

with open(LABEL_PATH, "rb") as f:
    classes = pickle.load(f)

print("[OK] Model + Labels loaded")
print(f"[OK] Number of classes: {len(classes)}")

# ==============================
# PARAMETERS (SAME AS TRAINING)
# ==============================
MAX_FRAMES = 12
IMG_SIZE = 96

# ==============================
# VIDEO PROCESSING
# ==============================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        return None

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    # Pad frames
    while len(frames) < MAX_FRAMES:
        frames.append(frames[-1])

    return np.array(frames)

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classes", methods=["GET"])
def get_classes():
    """Get all supported classes"""
    return jsonify({
        "total": len(classes),
        "classes": [cls.replace('_', ' ').title() for cls in classes]
    })

# ==============================
# PREDICTION API
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save temp file
    temp_path = "temp_video.mp4"
    file.save(temp_path)

    try:
        # Process video
        data = process_video(temp_path)

        if data is None:
            return jsonify({"error": "Invalid video or could not read frames"}), 400

        data = np.expand_dims(data, axis=0)

        # Prediction
        preds = model.predict(data, verbose=0)[0]

        top_idx = np.argmax(preds)
        confidence = float(preds[top_idx])

        predicted_class = classes[top_idx]

        # Get top 3 predictions
        top_3_idx = np.argsort(preds)[-3:][::-1]
        top_3 = [
            {
                "class": classes[idx].replace('_', ' ').title(),
                "confidence": float(preds[idx])
            }
            for idx in top_3_idx
        ]

        # Confidence check
        if confidence < 0.6:
            status = "low_confidence"
        else:
            status = "confident"

        result = {
            "sign_name": predicted_class.replace('_', ' ').title(),
            "confidence": f"{confidence:.2%}",
            "confidence_score": round(confidence * 100, 2),
            "status": status,
            "top_3": top_3
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    print("\nSignBERT Flask Server - 50 Classes")
    print("="*50)
    print(f"Total Classes: {len(classes)}")
    print("="*50)
    print("\nStarting server on port 7860 (Hugging Face Spaces)")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=False, host="0.0.0.0", port=7860)
