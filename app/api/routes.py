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
import RPi.GPIO as GPIO
import atexit
import signal
import sys
import requests

# Global variables
picam2 = None
net = None
INPUT_WIDTH = 960
INPUT_HEIGHT = 960
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Bottle points (default)
BOTTLE_POINTS = {"small": 10, "medium": 20, "large": 30}

# Fetch bottle points from Azure service
AZURE_POINTS_URL = "http://4.157.140.60:8000/api/bottles/public/points-info"


def fetch_bottle_points():
    global BOTTLE_POINTS
    try:
        resp = requests.get(AZURE_POINTS_URL, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            BOTTLE_POINTS = {
                "small": data.get("smallBottlePoints", 10),
                "medium": data.get("mediumBottlePoints", 20),
                "large": data.get("largeBottlePoints", 30),
            }
            print(f"Bottle points updated: {BOTTLE_POINTS}")
        else:
            print(f"Failed to fetch bottle points: {resp.status_code}")
    except Exception as e:
        print(f"Failed to fetch bottle points: {str(e)}")


# Fetch bottle points at startup
fetch_bottle_points()

# GPIO pin number
SERVO_PIN = 14

# Set GPIO mode and start PWM (only once)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)


def cleanup_gpio():
    pwm.stop()
    GPIO.cleanup()


# When the program exits, clean up the GPIO (only once)
atexit.register(cleanup_gpio)


# KeyboardInterrupt
def signal_handler(sig, frame):
    print("Exiting, cleaning up GPIO...")
    cleanup_gpio()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def control_servo():
    try:
        # 0 degree
        duty = 0 / 18 + 2
        GPIO.output(SERVO_PIN, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(1)
        GPIO.output(SERVO_PIN, False)
        pwm.ChangeDutyCycle(0)

        # Wait
        time.sleep(1)

        # 100 degree
        duty = 100 / 18 + 2
        GPIO.output(SERVO_PIN, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(1)
        GPIO.output(SERVO_PIN, False)
        pwm.ChangeDutyCycle(0)

        # Wait
        time.sleep(1)

        # First, return to 0 degree
        duty = 0 / 18 + 2
        GPIO.output(SERVO_PIN, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(1)
        GPIO.output(SERVO_PIN, False)
        pwm.ChangeDutyCycle(0)

        # Wait
        time.sleep(1)

        # Second, return to 0 degree
        duty = 0 / 18 + 2
        GPIO.output(SERVO_PIN, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(1)
        GPIO.output(SERVO_PIN, False)
        pwm.ChangeDutyCycle(0)

    except Exception as e:
        print(f"Servo motor control error: {str(e)}")


def cleanup_camera():
    global picam2
    try:
        if picam2 is not None:
            picam2.stop()
            picam2.close()
            picam2 = None
            time.sleep(2)  # Wait for the camera to fully close
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

        # Load YOLOv8 model with OpenCV
        net = cv2.dnn.readNetFromONNX(weights_path)

        # If CUDA is available, activate it
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU on Raspberry Pi

        print("Model successfully loaded")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        net = None
        return False


def process_image(image):
    original_height, original_width = image.shape[:2]
    print(f"Input image shape: {original_width}x{original_height}")

    # Prepare the image for the model
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)

    # Calculate scale factors between original image and model input
    scale_x = original_width / INPUT_WIDTH
    scale_y = original_height / INPUT_HEIGHT

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

                # Convert to pixel coordinates and scale to original image size
                x1 = int((x - w / 2) * INPUT_WIDTH * scale_x)
                y1 = int((y - h / 2) * INPUT_HEIGHT * scale_y)
                x2 = int((x + w / 2) * INPUT_WIDTH * scale_x)
                y2 = int((y + h / 2) * INPUT_HEIGHT * scale_y)

                # Ensure coordinates are within bounds
                x1 = max(0, min(x1, original_width - 1))
                y1 = max(0, min(y1, original_height - 1))
                x2 = max(0, min(x2, original_width - 1))
                y2 = max(0, min(y2, original_height - 1))

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
                    x = x / INPUT_WIDTH
                    y = y / INPUT_HEIGHT
                    w = w / INPUT_WIDTH
                    h = h / INPUT_HEIGHT

                # Scale to original image dimensions - this is the key correction
                x1 = int((x - w / 2) * INPUT_WIDTH * scale_x)
                y1 = int((y - h / 2) * INPUT_HEIGHT * scale_y)
                box_width = int(w * INPUT_WIDTH * scale_x)
                box_height = int(h * INPUT_HEIGHT * scale_y)

                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, original_width - 1))
                y1 = max(0, min(y1, original_height - 1))
                box_width = min(box_width, original_width - x1)
                box_height = min(box_height, original_height - y1)

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
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],  # Changed to [x1, y1, x2, y2] for clarity
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


# Start with camera and model
initialize_camera()
load_model()


@bp.route("/capture", methods=["GET"])
def capture():
    global picam2

    if picam2 is None or not initialize_camera():
        return jsonify({"success": False, "error": "Camera initialization failed"}), 500

    try:
        # Capture image
        image = picam2.capture_array()

        # Convert to PIL Image (already in RGB format)
        pil_image = Image.fromarray(image)

        # Convert to Base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({"success": True, "image": img_str})

    except Exception as e:
        print(f"Capture error: {str(e)}")
        cleanup_camera()  # Clean up the camera in case of error
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

        # Visualize detections
        if detections and len(detections) > 0:
            visualize_detections(image_np, detections, save_path="debug_detection.jpg")
            # Wait 3 seconds and then control the servo
            time.sleep(5)
            control_servo()

        # Calculate bottle counts
        bottle_counts = {"small": 0, "medium": 0, "large": 0}

        total_points = 0
        if detections:
            for det in detections:
                height = det["bbox"][3] - det["bbox"][1]  # y2 - y1
                if height < 450:  # Small bottle
                    bottle_counts["small"] += 1
                elif height < 600:  # Medium bottle
                    bottle_counts["medium"] += 1
                else:  # Large bottle
                    bottle_counts["large"] += 1
                total_points += det["points"]

        print(f"Final total points: {total_points}")
        print(f"Bottle counts: {bottle_counts}")

        # Tespit olmasa bile başarılı yanıt dön
        if total_points > 0:
            qr_data = generate_qr_code(total_points, bottle_counts)
        else:
            qr_data = "Something went wrong! No bottles detected in the container, please check the container. If you are experiencing a problem, please contact the support team!"

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

    print(f"Calculating points for height: {height}")

    if height < 100:  # Too small, likely false positive
        points = 0
        size = "too small - rejected"
    elif height < 450:  # Small bottle
        points = BOTTLE_POINTS["small"]
        size = "small"
    elif height < 600:  # Medium bottle
        points = BOTTLE_POINTS["medium"]
        size = "medium"
    else:  # Large bottle
        points = BOTTLE_POINTS["large"]
        size = "large"

    print(f"Bottle size: {size}, Points: {points}")
    return points


def visualize_detections(image, detections, save_path=None):
    """
    Draw bounding boxes on the image for detected bottles.
    The coordinates should already be scaled to the original image size from process_image.
    """
    vis_image = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]  # Already scaled to the original image
        conf = det["confidence"]
        points = det["points"]
        height = y2 - y1

        # Select color based on size
        if height < 300:
            color = (255, 0, 0)  # Blue - Small
            label_size = "Small"
        elif height < 450:
            color = (0, 255, 255)  # Yellow - Medium
            label_size = "Medium"
        else:
            color = (0, 0, 255)  # Red - Large
            label_size = "Large"

        # Debug output of the actual box coordinates
        print(f"Drawing box at: x1={x1}, y1={y1}, x2={x2}, y2={y2}, height={height}")

        # Fill the box inside with semi-transparent
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        alpha = 0.2  # Saydamlık
        cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)

        # Draw border
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{label_size} | {conf:.2f} | {points}pts"
        cv2.putText(
            vis_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    return vis_image


def generate_qr_code(points, bottle_counts):
    """Generate QR code in the desired format"""
    qr_data = {
        "containerId": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "points": points,
        "numberOfSmallBottles": bottle_counts.get("small", 0),
        "numberOfMediumBottles": bottle_counts.get("medium", 0),
        "numberOfLargeBottles": bottle_counts.get("large", 0),
    }

    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(json.dumps(qr_data))
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
