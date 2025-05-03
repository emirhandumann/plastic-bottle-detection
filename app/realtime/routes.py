from flask import render_template, jsonify, request, send_file
from app.realtime import bp
import cv2
import numpy as np
import threading
import time
import os
from picamera2 import Picamera2
import qrcode
from PIL import Image
import io

# Global değişkenler
detection_thread = None
is_processing = False
picam2 = None
net = None
all_bottles = []  # Tüm tespit edilen şişelerin merkezleri
bottle_count = 0
last_frame = None

# Model parametreleri
INPUT_WIDTH = 960
INPUT_HEIGHT = 960
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.4
DETECTION_INTERVAL = 2  # Saniye
IOU_THRESHOLD = 0.3  # Aynı şişe olup olmadığını anlamak için


def cleanup_camera():
    global picam2
    try:
        if picam2 is not None:
            picam2.stop()
            picam2.close()
            picam2 = None
            time.sleep(2)
    except Exception as e:
        print(f"Kamera temizleme hatası: {str(e)}")


def initialize_camera():
    global picam2
    try:
        cleanup_camera()
        time.sleep(2)
        picam2 = Picamera2()
        preview_config = picam2.create_preview_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )
        picam2.configure(preview_config)
        try:
            picam2.start(show_preview=False)
            time.sleep(3)
            print("Kamera başarıyla başlatıldı")
            return True
        except Exception as start_error:
            print(f"Kamera başlatma hatası: {str(start_error)}")
            cleanup_camera()
            return False
    except Exception as e:
        print(f"Kamera başlatma hatası: {str(e)}")
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
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {weights_path}")
        net = cv2.dnn.readNetFromONNX(weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Model başarıyla yüklendi")
        return True
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        net = None
        return False


def iou(boxA, boxB):
    # box: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def process_image(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward()
    boxes = []
    confidences = []
    if outputs.shape[1] == 84:
        for i in range(outputs.shape[2]):
            classes_scores = outputs[0, 4:, i]
            max_score = np.max(classes_scores)
            if max_score > CONFIDENCE_THRESHOLD:
                x, y, w, h = outputs[0, 0:4, i]
                x1 = int((x - w / 2) * width)
                y1 = int((y - h / 2) * height)
                x2 = int((x + w / 2) * width)
                y2 = int((y + h / 2) * height)
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width <= 1 or box_height <= 1:
                    continue
                boxes.append([x1, y1, box_width, box_height])
                confidences.append(float(max_score))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    final_boxes = [boxes[i] for i in indices]
    return final_boxes


def detection_loop():
    global is_processing, bottle_count, all_bottles, last_frame
    while is_processing:
        try:
            frame = picam2.capture_array()
            last_frame = frame.copy()
            detected_boxes = process_image(frame)
            new_bottles = 0
            for box in detected_boxes:
                is_new = True
                for old_box in all_bottles:
                    if iou(box, old_box) > IOU_THRESHOLD:
                        is_new = False
                        break
                if is_new:
                    all_bottles.append(box)
                    new_bottles += 1
            bottle_count = len(all_bottles)
            print(f"[DEBUG] Toplam şişe: {bottle_count}")
            time.sleep(DETECTION_INTERVAL)
        except Exception as e:
            print(f"[DEBUG] detection_loop hatası: {str(e)}")
            time.sleep(1)


def generate_qr_code(points, bottle_count):
    data = f"Puan: {points}, Şişe: {bottle_count}"
    qr = qrcode.make(data)
    qr_path = os.path.join("app", "static", "qr.png")
    qr.save(qr_path)
    return "/static/qr.png"


@bp.route("/")
def realtime():
    return render_template("realtime.html")


@bp.route("/start_detection", methods=["POST"])
def start_detection():
    global is_processing, detection_thread, bottle_count, all_bottles
    if not is_processing:
        if initialize_camera() and load_model():
            is_processing = True
            bottle_count = 0
            all_bottles = []
            detection_thread = threading.Thread(target=detection_loop)
            detection_thread.daemon = True
            detection_thread.start()
            return jsonify({"status": "success", "message": "Tespit başlatıldı"})
        else:
            return jsonify(
                {"status": "error", "message": "Kamera veya model başlatılamadı"}
            )
    else:
        return jsonify({"status": "error", "message": "Tespit zaten çalışıyor"})


@bp.route("/stop_detection", methods=["POST"])
def stop_detection():
    global is_processing
    is_processing = False
    cleanup_camera()
    return jsonify({"status": "success", "message": "Tespit durduruldu"})


@bp.route("/get_bottle_count")
def get_bottle_count():
    global bottle_count
    return jsonify({"bottle_count": bottle_count})


@bp.route("/generate_qr")
def generate_qr():
    global bottle_count
    points = bottle_count * 10  # Örnek puanlama
    qr_url = generate_qr_code(points, bottle_count)
    return jsonify({"qr_url": qr_url, "points": points, "bottle_count": bottle_count})


@bp.route("/last_frame")
def last_frame_img():
    global last_frame
    if last_frame is not None:
        _, buffer = cv2.imencode(".jpg", last_frame)
        return send_file(io.BytesIO(buffer.tobytes()), mimetype="image/jpeg")
    else:
        img = Image.new("RGB", (640, 480), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
