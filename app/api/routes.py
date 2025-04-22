from flask import jsonify, request
from app.api import bp
import cv2
import numpy as np
import base64
import qrcode
import io
import os
from PIL import Image
import time
from picamera2 import Picamera2

# Global değişkenler
picam2 = None
net = None
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45


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
        cleanup_camera()  # Önceki kamera bağlantısını temizle
        time.sleep(2)  # Daha uzun bir bekleme süresi ekleyin

        picam2 = Picamera2()
        # Kamera konfigürasyonunu basitleştirelim
        preview_config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}  # Daha düşük çözünürlük
        )
        picam2.configure(preview_config)

        try:
            picam2.start(show_preview=False)  # preview'i kapatın
            time.sleep(3)  # Kameranın başlaması için daha uzun bekleyin
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
            "exp1",
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

    # Görüntüyü hazırla
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)

    # Çıktıları al
    outputs = net.forward()

    # Sonuçları işle
    detections = []
    for detection in outputs[0]:
        confidence = detection[4]
        if confidence > CONFIDENCE_THRESHOLD:
            x = detection[0]
            y = detection[1]
            w = detection[2]
            h = detection[3]

            # Koordinatları orijinal görüntü boyutuna ölçekle
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            # Şişe boyutuna göre puan hesapla
            bottle_height = y2 - y1
            points = calculate_points(bottle_height)

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confidence),
                    "class": 0,  # Sadece şişe sınıfı var
                    "points": points,
                }
            )

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

        # Process image and get detections
        detections = process_image(image_np)

        # Generate QR code
        qr_data = generate_qr_code(sum(d["points"] for d in detections))

        return jsonify({"success": True, "detections": detections, "qr_code": qr_data})

    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 400


def calculate_points(height):
    """Şişe boyutuna göre puan hesapla"""
    if height < 100:
        return 10  # Küçük şişe
    elif height < 200:
        return 20  # Orta şişe
    else:
        return 30  # Büyük şişe


def generate_qr_code(points):
    """Puan bilgisini içeren QR kod oluştur"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(f"points:{points}")
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
