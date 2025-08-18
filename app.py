from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # enable CORS

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Lazy load model
model = None

# TensorFlow Lite model conversion
def convert_to_tflite():
    MODEL_PATH = os.path.join("model", "coloboma_detector.h5")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = os.path.join("model", "coloboma_detector.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to: {tflite_path}")

def load_model_once():
    global model
    if model is None:
        MODEL_PATH = os.path.join("model", "coloboma_detector.h5")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    mdl = load_model_once()
    img = Image.open(image_path).convert("RGB").resize((299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = mdl.predict(img_array)[0][0]

    if prediction >= 0.5:
        is_coloboma = False
        confidence = prediction * 100
    else:
        is_coloboma = True
        confidence = (1 - prediction) * 100

    return is_coloboma, round(float(confidence), 2)

@app.route("/predict", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        is_coloboma, confidence = predict_image(filepath)

        return jsonify({
            "isColoboma": is_coloboma,
            "confidence": confidence
        })

    return jsonify({"error": "Invalid file type"}), 400

@app.route("/convert-to-tflite", methods=["POST"])
def convert_model():
    try:
        convert_to_tflite()
        return jsonify({"message": "Model successfully converted to TensorFlow Lite format"})
    except Exception as e:
        return jsonify({"error": f"Failed to convert model: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)