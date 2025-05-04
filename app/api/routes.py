from flask import jsonify, request, Response
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
import threading
from PIL import Image
from picamera2 import Picamera2

# Global değişkenler
picam2 = None
net = None
INPUT_WIDTH = 960
INPUT_HEIGHT = 960
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.4
detection_active = False
detection_thread = None
current_detections = []
bottle_counts = {"small": 0, "medium": 0, "large": 0}
total_points = 0
last_frame = None
processing_lock = threading.Lock()


def cleanup_camera():
    global picam2
    try:
        if picam2 is not None:
            picam2.stop()
            picam2.close()
            picam2 = None
            time.sleep(2)
    except Exception as e:
        print(f"Camera cleanup error: {str(e)}")


def initialize_camera():
    global picam2
    try:
        cleanup_camera()
        time.sleep(2)

        picam2 = Picamera2()
        # Daha düşük çözünürlüklü ve daha yüksek FPS ayarları
        preview_config = picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        picam2.configure(preview_config)

        try:
            picam2.start(show_preview=False)
            time.sleep(3)
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
        # YOLOv8n (nano) gibi daha hafif bir model kullanmayı düşünün
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
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print("Model successfully loaded")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        net = None
        return False


def process_image(image):
    print(f"[LOG] process_image: input shape: {image.shape}")
    height, width = image.shape[:2]

    # Modelin girdi boyutlarına göre resmi küçült
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)

    # Get outputs
    outputs = net.forward()
    print(f"[LOG] process_image: model output shape: {outputs.shape}")

    # Process results
    boxes = []
    confidences = []
    class_ids = []

    # YOLOv8 output processing - modelin çıktı formatına göre uyarlayın
    rows = outputs.shape[2]
    for i in range(rows):
        confidence = float(outputs[0, 4, i])

        # Confidence threshold
        if confidence > CONFIDENCE_THRESHOLD:
            # Get normalized coordinates
            x = float(outputs[0, 0, i])
            y = float(outputs[0, 1, i])
            w = float(outputs[0, 2, i])
            h = float(outputs[0, 3, i])

            # Convert to pixel coordinates
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            box_width = int(w * width)
            box_height = int(h * height)

            # Ensure coordinates are within bounds
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

    # Apply non-max suppression
    detections = []
    if len(boxes) > 0:
        try:
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
            )

            # Check if indices is numpy array or list
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()

            # Process each valid detection
            for i in indices:
                box = boxes[i]
                x1, y1 = box[0], box[1]
                w, h = box[2], box[3]
                x2, y2 = x1 + w, y1 + h

                # Calculate bottle height
                bottle_height = h

                # Skip detections with tiny height
                if bottle_height < 10:
                    continue

                points = calculate_points(bottle_height)

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(confidences[i]),
                        "class": class_ids[i],
                        "points": points,
                    }
                )

        except Exception as e:
            print(f"Error during NMS: {str(e)}")

    print(f"[LOG] process_image: tespit edilen nesne sayısı: {len(boxes)}")
    print(f"[LOG] process_image: dönen detection sayısı: {len(detections)}")
    return detections


def calculate_points(height):
    """Calculate points based on bottle height"""
    # Based on your terminal output, bottles appear to have heights around 400-500 pixels
    if height < 100:  # Too small, likely false positive
        points = 0
    elif height < 300:  # Small bottle
        points = 10
    elif height < 450:  # Medium bottle
        points = 20
    else:  # Large bottle
        points = 30

    return points


def visualize_detections(image, detections):
    """Draw bounding boxes on the image"""
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

    return vis_image


def generate_qr_code():
    """Generate QR code with points, bottle counts and timestamp"""
    global total_points, bottle_counts

    qr_data = {
        "type": "green_earn_points",
        "points": total_points,
        "bottle_counts": bottle_counts,
        "timestamp": int(time.time()),  # Unix timestamp
        "version": "1.0",
        "checksum": hashlib.sha256(
            f"{total_points}:{int(time.time())}:green_earn_secret".encode()
        ).hexdigest(),
    }

    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(json.dumps(qr_data))
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def detection_loop():
    global detection_active, current_detections, total_points, bottle_counts, last_frame

    print("Starting detection loop")

    # Bir önceki tespitleri sıfırla
    current_detections = []
    total_points = 0
    bottle_counts = {"small": 0, "medium": 0, "large": 0}

    # Otomatik tespit özelliği: Belirli aralıklarla görüntü al ve şişeleri tespit et
    while detection_active:
        try:
            if picam2 is None:
                if not initialize_camera():
                    print("Camera initialization failed in detection loop")
                    time.sleep(1)
                    continue

            # Görüntü yakala
            frame = picam2.capture_array()
            print(f"[LOG] detection_loop: alınan frame shape: {frame.shape}")
            with processing_lock:
                last_frame = frame.copy()

            # Tespit işlemi
            detections = process_image(frame)
            print(
                f"[LOG] detection_loop: tespit edilen detection sayısı: {len(detections)}"
            )

            # Şişeleri incele ve sınıflandır
            if detections:
                with processing_lock:
                    # Yeni şişeleri ara
                    for det in detections:
                        # İzleme algoritması için gerçek tespit mantığı burada geliştirilebilir
                        # Burada basit bir yaklaşım kullanıyoruz

                        x1, y1, x2, y2 = det["bbox"]
                        height = y2 - y1

                        # Şişe boyutunu hesapla
                        if height < 300:  # Small bottle
                            bottle_counts["small"] += 1
                        elif height < 450:  # Medium bottle
                            bottle_counts["medium"] += 1
                        else:  # Large bottle
                            bottle_counts["large"] += 1

                        total_points += det["points"]

                    current_detections = detections

            # FPS hızı ayarlanabilir (Raspberry Pi CPU kullanımı için)
            time.sleep(5)  # 10 saniye bekle

        except Exception as e:
            print(f"Error in detection loop: {str(e)}")
            time.sleep(1)


# Başlangıçta kamera ve modeli yükle
initialize_camera()
load_model()


@bp.route("/start_detection", methods=["POST"])
def start_detection():
    global detection_active, detection_thread, current_detections, total_points, bottle_counts

    if detection_active:
        return jsonify({"success": False, "error": "Detection already active"}), 400

    try:
        # Kamera kontrol ve model kontrol
        if picam2 is None and not initialize_camera():
            return (
                jsonify({"success": False, "error": "Camera initialization failed"}),
                500,
            )

        if net is None and not load_model():
            return jsonify({"success": False, "error": "Model loading failed"}), 500

        # Değerleri sıfırla
        current_detections = []
        total_points = 0
        bottle_counts = {"small": 0, "medium": 0, "large": 0}

        # Algılama döngüsünü başlat
        detection_active = True
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()

        return jsonify({"success": True, "message": "Detection started"})

    except Exception as e:
        print(f"Error starting detection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/stop_detection", methods=["POST"])
def stop_detection():
    global detection_active, detection_thread, current_detections, total_points, bottle_counts

    if not detection_active:
        return jsonify({"success": False, "error": "Detection not active"}), 400

    try:
        # Algılama döngüsünü durdur
        detection_active = False

        # İş parçacığının bitmesini bekle
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=3.0)

        # QR kodunu oluştur
        qr_data = generate_qr_code() if total_points > 0 else None

        # Sonuçları döndür
        result = {
            "success": True,
            "detections": current_detections,
            "bottle_counts": bottle_counts,
            "total_points": total_points,
            "qr_code": qr_data,
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error stopping detection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/detection_status", methods=["GET"])
def detection_status():
    global detection_active, current_detections, total_points, bottle_counts, last_frame

    try:
        # Son görüntüyü ve tespitleri döndür
        with processing_lock:
            if last_frame is not None:
                # Tespitleri görüntüye işle
                if current_detections:
                    frame_with_detections = visualize_detections(
                        last_frame, current_detections
                    )
                else:
                    frame_with_detections = last_frame

                # Base64'e çevir
                pil_image = Image.fromarray(frame_with_detections)
                buffered = io.BytesIO()
                pil_image.save(
                    buffered, format="JPEG", quality=70
                )  # Kaliteyi düşürerek veri miktarını azalt
                img_str = base64.b64encode(buffered.getvalue()).decode()
            else:
                img_str = None

        return jsonify(
            {
                "success": True,
                "active": detection_active,
                "current_frame": img_str,
                "bottle_counts": bottle_counts,
                "total_points": total_points,
                "detection_count": len(current_detections) if current_detections else 0,
            }
        )

    except Exception as e:
        print(f"Error getting detection status: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/video_feed", methods=["GET"])
def video_feed():
    """MJPEG video akışı oluştur"""

    def generate():
        global last_frame, current_detections
        while True:
            # Lock ile son frame'e erişim
            with processing_lock:
                if last_frame is not None:
                    if current_detections:
                        # Tespitleri görüntüye işle
                        frame = visualize_detections(last_frame, current_detections)
                    else:
                        frame = last_frame.copy()

                    # JPEG'e dönüştür
                    ret, jpeg = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    )
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + jpeg.tobytes()
                            + b"\r\n"
                        )

            # FPS ayarlaması
            time.sleep(0.1)  # ~10 FPS

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
