import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from  pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
from werkzeug.utils import secure_filename
import datetime
import base64
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Connect to MongoDB
client = MongoClient(os.getenv("MONGODB_DB"))
db = client["coloboma"]
userCollection = db["users"]
historyCollection = db["history"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/coloboma_detector.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((299, 299))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if prediction >= 0.5:
        return False, float(prediction * 100)   # Normal
    else:
        return True, float((1 - prediction) * 100)  # Coloboma

@app.route("/predict", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400


    user_id = request.form.get("id")
    if not user_id:
        return jsonify({"error": "User ID required"}), 400

    # Unique filename per upload
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)

    file.save(filepath)

    try:
        is_coloboma, confidence = predict_image(filepath)
    except Exception:
        return jsonify({"error": "Prediction failed"}), 500

    # Convert image to base64 for storage
    with open(filepath, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Store in database with base64 image
    historyCollection.insert_one({
        "userId": user_id,
        "imageName": unique_name,
        "imageData": encoded_image,  # Store base64 encoded image
        "isColoboma": is_coloboma,
        "confidence": round(confidence, 2),
        "createdAt": datetime.datetime.utcnow()
    })

    return jsonify({
        "isColoboma": is_coloboma,
        "confidence": round(confidence, 2)
    }), 200

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"success": False, "message": "Invalid JSON payload"}), 400

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password required"}), 400
    
    user = userCollection.find_one({"email": email})

    if not user:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    if not check_password_hash(user["password"], password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    return jsonify({
        "success": True,
        "name": user.get("name", ""),
        "id": str(user["_id"]),
        "message": "Login successful"
    }), 200
    
@app.route("/register", methods = ["POST"])
def register():
    data = request.get_json()
    userName = data.get("name")
    userEmail = data.get("email")
    password = data.get("password")
    if not userEmail or not password:
        return jsonify({"success": False, "message": "Email and password required"}), 400
    
    if userCollection.find_one({"email" : userEmail}):
        return jsonify({"success": False, "message": "User already exists"}), 409
    
    hashed_password = generate_password_hash(password)
    userCollection.insert_one({"name" : userName, "email": userEmail, "password": hashed_password})

    return jsonify({"success": True, "message": "User registered successfully", "id": str(userCollection.find_one({"email": userEmail})["_id"])}), 201

@app.route("/history/<id>", methods=["GET"])
def medicalHistory(id):
    user_id = id
    if not user_id:
        return jsonify({"error": "User ID required"}), 400

    records = historyCollection.find({"userId": user_id}).sort("createdAt", -1)
    history = []
    for record in records:
        history.append({
            "imageName": record["imageName"],
            "imageData": record.get("imageData", ""),  # Base64 image data
            "isColoboma": record["isColoboma"],
            "confidence": record["confidence"],
            "createdAt": record["createdAt"].isoformat() + "Z"
        })
    print(history)

    return jsonify({"history": history}), 200

if __name__ == "__main__":
    app.run(debug=True)
