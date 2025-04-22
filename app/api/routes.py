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
CONFIDENCE_THRESHOLD = 0.70
NMS_THRESHOLD = 0.2


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
    print(f"Input image size: {width}x{height}")

    # Görüntüyü hazırla
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)

    # Çıktıları al
    outputs = net.forward()
    print(f"Network output shape: {outputs.shape}")

    # Sonuçları işle
    detections = []
    detection_count = 0

    # Tüm tespitleri topla
    boxes = []
    confidences = []

    # Her bir tespit için
    for detection in outputs[0]:
        # Sınıf olasılıklarını al (5. elemandan sonrası)
        scores = detection[5:]
        # En yüksek olasılığa sahip sınıfı bul
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Güven eşiğini kontrol et
        if confidence > CONFIDENCE_THRESHOLD:
            # Normalize edilmiş koordinatları al
            x = detection[0]
            y = detection[1]
            w = detection[2]
            h = detection[3]

            # Koordinatları orijinal görüntü boyutuna ölçekle
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            # Görüntü sınırlarını kontrol et
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Geçerli bir kutu mu kontrol et
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(confidence))

    # Non-maximum suppression uygula
    detections = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
        )

        for i in indices:
            idx = i
            box = boxes[idx]
            x1 = box[0]
            y1 = box[1]
            w = box[2]
            h = box[3]
            x2 = x1 + w
            y2 = y1 + h

            detection_count += 1
            bottle_height = y2 - y1
            points = calculate_points(bottle_height)

            print(f"Detection {detection_count}:")
            print(f"  - Confidence: {confidences[idx]*100:.2f}%")
            print(f"  - Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  - Bottle height: {bottle_height} pixels")
            print(f"  - Points awarded: {points}")

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidences[idx],
                    "class": 0,  # Sadece şişe sınıfı var
                    "points": points,
                }
            )

    total_points = sum(d["points"] for d in detections)
    print(f"Total detections: {len(detections)}")
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

        # Generate QR code
        total_points = sum(d["points"] for d in detections)
        print(f"Final total points: {total_points}")
        qr_data = generate_qr_code(total_points)

        return jsonify(
            {
                "success": True,
                "detections": detections,
                "qr_code": qr_data,
                "debug_info": {
                    "num_detections": len(detections),
                    "individual_points": [d["points"] for d in detections],
                    "total_points": total_points,
                },
            }
        )

    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 400


def calculate_points(height):
    """Şişe boyutuna göre puan hesapla"""
    print(f"Calculating points for height: {height}")
    if height < 5000:
        points = 10  # Küçük şişe
        size = "small"
    elif height < 10000:
        points = 20  # Orta şişe
        size = "medium"
    else:
        points = 30  # Büyük şişe
        size = "large"
    print(f"Bottle size: {size}, Points: {points}")
    return points


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
