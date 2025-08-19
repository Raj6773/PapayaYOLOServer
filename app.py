from flask import Flask, request, send_file, jsonify
from YoloDetector import YoloDetector
import os
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model (TorchScript)
detector = YoloDetector(model_path="best.torchscript")

@app.route("/")
def index():
    return "YOLOv8 TorchScript Server is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Resize to 640x640 for memory safety
    try:
        image = Image.open(file_path)
        image = image.resize((640, 640))
        image.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    # Output image path
    output_path = os.path.join(UPLOAD_FOLDER, "output_" + file.filename)

    try:
        # Detect and draw bounding boxes
        detections = detector.detect_and_draw(file_path, save_path=output_path)
        
        # Send output image with correct MIME type
        output_image = Image.open(output_path)
        mimetype = f"image/{output_image.format.lower()}"
        return send_file(output_path, mimetype=mimetype)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
