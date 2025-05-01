from flask import jsonify, request
from app.api import bp
import cv2
import numpy as np
import base64
import qrcode
import io
import os
import json
import time
import hashlib
from PIL import Image
from picamera2 import Picamera2

# Global değişkenler
picam2 = None
net = None
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.4


def cleanup_camera():
    global picam2
    try:
        if picam2 is not None:
            picam2.stop()
            picam2.close()
            picam2 = None
            time.sleep(2)  # Kameranın tamamen kapanması için bekle
    except Exception as e:
        print(f"Camera cleanup error: {str(e)}")


def initialize_camera():
    global picam2
    try:
        cleanup_camera()
        time.sleep(2)  # Longer wait after cleanup

        picam2 = Picamera2()
        # Simplified camera configuration
        preview_config = picam2.create_preview_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )
        picam2.configure(preview_config)

        try:
            picam2.start(show_preview=False)
            time.sleep(3)  # Longer wait after start
            print("Camera initialized successfully")
            return True
        except Exception as start_error:
            print(f"Camera start error: {str(start_error)}")
            cleanup_camera()
            return False

    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        picam2 = None
        return False


def load_model():
    global net
    try:
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "plastic_bottle_detection",
            "exp2",
            "weights",
            "best.onnx",
        )

        print(f"Weights path: {weights_path}")
        print(f"Model file exists: {os.path.exists(weights_path)}")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model file not found at {weights_path}")

        # YOLOv8 modelini OpenCV ile yükle
        net = cv2.dnn.readNetFromONNX(weights_path)

        # CUDA kullanılabilirse aktif et
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Raspberry Pi'de CPU kullan

        print("Model successfully loaded")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        net = None
        return False


def process_image(image):
    height, width = image.shape[:2]
    print(f"Input image shape: {width}x{height}")

    # Prepare the image
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)

    # Get outputs
    outputs = net.forward()
    print(f"Network output shape: {outputs.shape}")

    # Process results
    detections = []
    detection_count = 0

    # Collect all detections
    boxes = []
    confidences = []
    class_ids = []

    print("\n=== Debug: All Detections ===")
    print(f"Model output shape: {outputs.shape}")

    # YOLOv8 output processing
    if outputs.shape[1] == 84:  # YOLOv8 standard output format (may vary)
        # Standard YOLOv8 output processing
        for i in range(outputs.shape[2]):
            classes_scores = outputs[0, 4:, i]
            max_score = np.max(classes_scores)

            if max_score > CONFIDENCE_THRESHOLD:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[0, 0:4, i]

                # Convert to pixel coordinates
                x1 = int((x - w / 2) * width)
                y1 = int((y - h / 2) * height)
                x2 = int((x + w / 2) * width)
                y2 = int((y + h / 2) * height)

                # Ensure coordinates are within bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                box_width = x2 - x1
                box_height = y2 - y1

                # Skip tiny or invalid boxes
                if box_width <= 1 or box_height <= 1:
                    continue

                boxes.append([x1, y1, box_width, box_height])
                confidences.append(float(max_score))
                class_ids.append(class_id)

                print(f"Debug - Detection {len(boxes)}:")
                print(f"  - Confidence: {max_score*100:.2f}%")
                print(f"  - Coordinates: x={x1}, y={y1}, w={box_width}, h={box_height}")

    elif outputs.shape[1] == 5:  # Your specific output format
        rows = outputs.shape[2]
        print(f"Number of potential detections: {rows}")

        # For each detection
        for i in range(rows):
            # Get confidence score
            confidence = float(outputs[0, 4, i])

            # Apply sigmoid if needed
            if confidence < 0 or confidence > 1:
                confidence = 1 / (1 + np.exp(-confidence))

            # Debug information
            if confidence > 0.2:  # Lower threshold for debugging
                print(f"Debug - Detection {i}:")
                print(f"  - Raw confidence: {confidence*100:.2f}%")
                print(
                    f"  - Raw coordinates: x={outputs[0, 0, i]}, y={outputs[0, 1, i]}, w={outputs[0, 2, i]}, h={outputs[0, 3, i]}"
                )

            # Check confidence threshold (use a slightly lower threshold here)
            if confidence > 0.25:  # Lower threshold for now
                # Get normalized coordinates
                x = float(outputs[0, 0, i])
                y = float(outputs[0, 1, i])
                w = float(outputs[0, 2, i])
                h = float(outputs[0, 3, i])

                # Validate coordinates (sometimes models output raw values, not normalized)
                if x > 1.0 or y > 1.0:  # Raw pixel values detected
                    # Convert to normalized coordinates
                    x = x / width
                    y = y / height
                    w = w / width
                    h = h / height

                # Scale to image dimensions
                x1 = int((x - w / 2) * width)
                y1 = int((y - h / 2) * height)
                box_width = int(w * width)
                box_height = int(h * height)

                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                box_width = min(box_width, width - x1)
                box_height = min(box_height, height - y1)

                # Skip tiny boxes
                if box_width <= 5 or box_height <= 5:
                    continue

                boxes.append([x1, y1, box_width, box_height])
                confidences.append(confidence)
                class_ids.append(0)  # Single class: bottle

                print(
                    f"Adding box: x1={x1}, y1={y1}, width={box_width}, height={box_height}, conf={confidence:.4f}"
                )

    else:
        print(f"Unrecognized output format: {outputs.shape}")

    print(f"\n=== Debug: Before NMS ===")
    print(f"Detections before NMS: {len(boxes)}")
    if boxes:
        print(f"Box coordinates example: {boxes[0]}")

    # Apply non-maximum suppression
    detections = []
    if len(boxes) > 0:
        try:
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
            )

            # Check if indices is numpy array or list
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()

            print(f"Detections after NMS: {len(indices)}")

            # Process each valid detection
            for i in indices:
                box = boxes[i]
                x1, y1 = box[0], box[1]
                w, h = box[2], box[3]
                x2, y2 = x1 + w, y1 + h

                # Calculate bottle height
                bottle_height = h

                # Skip detections with zero or tiny height
                if bottle_height < 10:
                    print(f"Skipping too small detection (height: {bottle_height})")
                    continue

                points = calculate_points(bottle_height)

                print(f"\nDetection {detection_count + 1}:")
                print(f"  - Confidence: {confidences[i]*100:.2f}%")
                print(f"  - Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"  - Bottle height: {bottle_height} pixels")
                print(f"  - Points awarded: {points}")

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(confidences[i]),
                        "class": class_ids[i],
                        "points": points,
                    }
                )
                detection_count += 1

        except Exception as e:
            print(f"Error during NMS: {str(e)}")

    total_points = sum(d["points"] for d in detections)
    print(f"\nFinal detection count: {len(detections)}")
    print(f"Total points: {total_points}")
    return detections


# Başlangıçta kamera ve modeli yükle
initialize_camera()
load_model()


@bp.route("/capture", methods=["GET"])
def capture():
    global picam2

    if picam2 is None or not initialize_camera():
        return jsonify({"success": False, "error": "Camera initialization failed"}), 500

    try:
        # Görüntü yakala
        image = picam2.capture_array()

        # PIL Image'e çevir (zaten RGB formatında)
        pil_image = Image.fromarray(image)

        # Base64'e çevir
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({"success": True, "image": img_str})

    except Exception as e:
        print(f"Capture error: {str(e)}")
        cleanup_camera()  # Hata durumunda kamerayı temizle
        return jsonify({"success": False, "error": str(e)}), 400


@bp.route("/detect", methods=["POST"])
def detect():
    global net

    if net is None and not load_model():
        return jsonify({"success": False, "error": "Model loading failed"}), 500

    try:
        # Get image from request
        data = request.get_json()
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        print("\n=== New Detection Request ===")
        print(f"Input image shape: {image_np.shape}")

        # Process image and get detections
        detections = process_image(image_np)

        # Şişe sayılarını hesapla
        bottle_counts = {"small": 0, "medium": 0, "large": 0}

        total_points = 0
        if detections:
            for det in detections:
                height = det["bbox"][3] - det["bbox"][1]  # y2 - y1
                if height < 300:  # Small bottle
                    bottle_counts["small"] += 1
                elif height < 450:  # Medium bottle
                    bottle_counts["medium"] += 1
                else:  # Large bottle
                    bottle_counts["large"] += 1
                total_points += det["points"]

        print(f"Final total points: {total_points}")
        print(f"Bottle counts: {bottle_counts}")

        # Tespit olmasa bile başarılı yanıt dön
        qr_data = (
            generate_qr_code(total_points, bottle_counts) if total_points > 0 else None
        )

        return jsonify(
            {
                "success": True,
                "detections": detections,
                "qr_code": qr_data,
                "debug_info": {
                    "num_detections": len(detections) if detections else 0,
                    "bottle_counts": bottle_counts,
                    "total_points": total_points,
                },
            }
        )

    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 400


def calculate_points(height):
    """Calculate points based on bottle height"""
    print(f"Calculating points for height: {height}")

    # Based on your terminal output, bottles appear to have heights around 400-500 pixels
    if height < 100:  # Too small, likely false positive
        points = 0
        size = "too small - rejected"
    elif height < 300:  # Small bottle
        points = 10
        size = "small"
    elif height < 450:  # Medium bottle
        points = 20
        size = "medium"
    else:  # Large bottle
        points = 30
        size = "large"

    print(f"Bottle size: {size}, Points: {points}")
    return points


def visualize_detections(image, detections, save_path="debug_detection.jpg"):
    """Draw bounding boxes on the image for debugging"""
    vis_image = image.copy()

    # Draw all boxes from the detection step
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        points = det["points"]

        # Draw rectangle
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display confidence and points
        label = f"{conf:.2f}, {points}pts"
        cv2.putText(
            vis_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save the visualized image
    cv2.imwrite(save_path, vis_image)
    print(f"Debug visualization saved to {save_path}")

    return vis_image


def generate_qr_code(points, bottle_counts):
    """Generate QR code with points, bottle counts and timestamp"""
    qr_data = {
        "type": "green_earn_points",
        "points": points,
        "bottle_counts": bottle_counts,
        "timestamp": int(time.time()),  # Unix timestamp
        "version": "1.0",
        "checksum": hashlib.sha256(
            f"{points}:{int(time.time())}:green_earn_secret".encode()
        ).hexdigest(),
    }

    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(json.dumps(qr_data))
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
