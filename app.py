from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image
import anthropic
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

app = Flask(__name__)

# YOLO model
model = YOLO("plastic_bottle_detection/exp1/weights/best.pt")

# Claude client
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)


def estimate_bottle_weight(image_np, bbox):
    """Claude 3 Vision API ile şişe ağırlığını tahmin et"""
    try:
        # Crop image within bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cropped_image = image_np[y1:y2, x1:x2]

        # Convert image to base64
        _, buffer = cv2.imencode(".jpg", cropped_image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        # Prompt to send to Claude
        prompt = """
        Analyze this plastic bottle image and estimate its approximate weight in grams.
        Consider the following factors:
        1. Bottle type (water, soda, etc.)
        2. Fill level percentage
        3. Dimensions (in pixels)

        Please provide only the estimated weight in grams as a number.
        """

        # Claude API call
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )

        # Extract weight value from response
        weight_text = response.content[0].text
        try:
            weight = float("".join(filter(str.isdigit, weight_text)))
            return weight
        except:
            return None

    except Exception as e:
        print(f"Weight estimation error: {str(e)}")
        return None


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image
        file = request.files["image"]

        # Convert image to numpy array
        image = Image.open(file.stream)
        image_np = np.array(image)

        # Predict with YOLO
        results = model(image_np)

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Estimate weight
                weight = estimate_bottle_weight(image_np, [x1, y1, x2, y2])

                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class": cls,
                        "estimated_weight": (
                            weight if weight is not None else "Tahmin edilemedi"
                        ),
                    }
                )

        return jsonify({"success": True, "detections": detections})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
