from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def suppress_duplicate_detections(detections, model_names, iou_threshold=0.15):
    """
    Suppress duplicate detections of the same class using NMS per class.
    
    Args:
        detections: YOLO detection boxes
        model_names: Class name mapping from model
        iou_threshold: IoU threshold for considering boxes as duplicates (0.15 = 15% overlap)
    
    Returns:
        List of unique product names after suppression
    """
    if len(detections) == 0:
        return []
    
    # Group detections by class
    class_detections = {}
    
    for i, box in enumerate(detections):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        
        if cls not in class_detections:
            class_detections[cls] = []
        
        class_detections[cls].append({
            'index': i,
            'conf': conf,
            'box': xyxy,
            'label': model_names[cls]
        })
    
    # Apply NMS per class
    final_products = []
    
    for cls, det_list in class_detections.items():
        # Sort by confidence (highest first)
        det_list.sort(key=lambda x: x['conf'], reverse=True)
        
        keep_indices = []
        
        for i, current_det in enumerate(det_list):
            should_keep = True
            
            # Check against all previously kept detections of the same class
            for kept_idx in keep_indices:
                kept_det = det_list[kept_idx]
                iou = calculate_iou(current_det['box'], kept_det['box'])
                
                # If IoU is above 15%, suppress this detection
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep_indices.append(i)
                final_products.append(current_det['label'])
    
    return final_products

@app.route("/")
def home():
    return jsonify({"message": "SmartCart YOLO Backend is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"message": "No image uploaded"}), 400
    
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    
    # Run YOLO inference
    results = model.predict(image, conf=0.0)
    detections = results[0].boxes
    
    # Apply duplicate suppression with 15% IoU threshold
    product_list = suppress_duplicate_detections(
        detections, 
        model.names, 
        iou_threshold=0.15  # Boxes with >15% overlap of same class = duplicates
    )
    
    return jsonify({
        "message": "Prediction successful",
        "products_detected": product_list,
        "total_count": len(product_list)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)