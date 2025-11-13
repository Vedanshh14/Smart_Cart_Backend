from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # allow frontend (React) to access this API

# Load your trained YOLO model
MODEL_PATH = "best.pt"  # <-- change this to your actual best.pt path
model = YOLO(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({"message": "SmartCart YOLO Backend is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"message": "No image uploaded"}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))

    # Run prediction
    results = model.predict(image, conf=0.10)  # you can tweak confidence threshold
    detections = results[0].boxes

    product_list = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        product_list.append(label)
    
    # Remove duplicates (optional)
    # product_list = list(set(product_list))

    return jsonify({
        "message": "Prediction successful",
        "products_detected": product_list
    })

if __name__ == "__main__":
    # Get PORT from environment variable (Railway provides this)
    # Default to 5001 for local development
    port = int(os.environ.get("PORT", 5001))
    
    # Use debug=False in production (Railway sets this automatically)
    debug = os.environ.get("FLASK_ENV") == "development"
    
    app.run(host="0.0.0.0", port=port, debug=debug)