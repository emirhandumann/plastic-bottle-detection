from flask import render_template, Response, jsonify, request
from app.realtime import bp
import cv2
import numpy as np
import threading
import time
import os
from picamera2 import Picamera2
from queue import Queue

# Global değişkenler
picam2 = None
net = None
frame_queue = Queue(maxsize=2)
processing_queue = Queue(maxsize=2)
is_processing = False
detection_thread = None

# Model parametreleri
INPUT_WIDTH = 960
INPUT_HEIGHT = 960
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.4

camera_lock = threading.Lock()


def cleanup_camera():
    global picam2
    with camera_lock:
        print("[DEBUG] cleanup_camera: başlıyor")
        try:
            if picam2 is not None:
                print("[DEBUG] cleanup_camera: picam2.stop() çağrılıyor")
                picam2.stop()
                print("[DEBUG] cleanup_camera: picam2.close() çağrılıyor")
                picam2.close()
                print("[DEBUG] cleanup_camera: picam2 = None atanıyor")
                picam2 = None
                print("[DEBUG] cleanup_camera: time.sleep(2) çağrılıyor")
                time.sleep(2)
        except Exception as e:
            print(f"[DEBUG] cleanup_camera: hata: {str(e)}")
        print("[DEBUG] cleanup_camera: bitti")


def initialize_camera():
    global picam2
    with camera_lock:
        print("[DEBUG] initialize_camera: başlıyor")
        try:
            cleanup_camera()
            print("[DEBUG] initialize_camera: cleanup_camera sonrası")
            time.sleep(2)
            picam2 = Picamera2()
            print("[DEBUG] initialize_camera: Picamera2 nesnesi oluşturuldu")
            preview_config = picam2.create_preview_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            print("[DEBUG] initialize_camera: preview_config oluşturuldu")
            picam2.configure(preview_config)
            print("[DEBUG] initialize_camera: configure sonrası")
            try:
                picam2.start(show_preview=False)
                print("[DEBUG] initialize_camera: start sonrası")
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


def process_frame(frame):
    if net is None:
        return [], frame

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward()

    detections = []
    boxes = []
    confidences = []
    class_ids = []

    if outputs.shape[1] == 84:
        for i in range(outputs.shape[2]):
            classes_scores = outputs[0, 4:, i]
            max_score = np.max(classes_scores)

            if max_score > CONFIDENCE_THRESHOLD:
                class_id = np.argmax(classes_scores)
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
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        confidence = confidences[i]
        class_id = class_ids[i]

        detections.append(
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": [int(x), int(y), int(w), int(h)],
            }
        )

        # Görselleştirme
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Bottle: {confidence:.2f}"
        cv2.putText(
            frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return detections, frame


def capture_frames():
    global picam2, is_processing
    while is_processing:
        try:
            frame = picam2.capture_array()
            print("[DEBUG] Frame alındı")
            if frame_queue.full():
                frame_queue.get()  # Eski kareyi at
            frame_queue.put(frame)
        except Exception as e:
            print(f"Kare yakalama hatası: {str(e)}")
            time.sleep(0.1)


def process_frames():
    global is_processing
    while is_processing:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                print("[DEBUG] Frame işleniyor")
                detections, processed_frame = process_frame(frame)
                if processing_queue.full():
                    processing_queue.get()
                processing_queue.put((detections, processed_frame))
        except Exception as e:
            print(f"Kare işleme hatası: {str(e)}")
            time.sleep(0.1)


def generate_frames():
    global is_processing
    while is_processing:
        try:
            if not processing_queue.empty():
                detections, frame = processing_queue.get()
                print("[DEBUG] Frame gönderiliyor (web)")
                ret, buffer = cv2.imencode(".jpg", frame)
                frame = buffer.tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
        except Exception as e:
            print(f"Frame oluşturma hatası: {str(e)}")
            time.sleep(0.1)


@bp.route("/")
def realtime():
    return render_template("realtime.html")


@bp.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@bp.route("/start_detection", methods=["POST"])
def start_detection():
    global is_processing, detection_thread, picam2
    print("[DEBUG] start_detection fonksiyonu çağrıldı")
    camera_ok = initialize_camera()
    print(f"[DEBUG] initialize_camera sonucu: {camera_ok}")
    model_ok = load_model()
    print(f"[DEBUG] load_model sonucu: {model_ok}")
    if not is_processing:
        if camera_ok and model_ok:
            is_processing = True
            print("[DEBUG] Thread'ler başlatılıyor")
            # Thread'leri başlat
            capture_thread = threading.Thread(target=capture_frames)
            process_thread = threading.Thread(target=process_frames)
            capture_thread.daemon = True
            process_thread.daemon = True
            capture_thread.start()
            process_thread.start()
            print("[DEBUG] Thread'ler başlatıldı")
            return jsonify({"status": "success", "message": "Tespit başlatıldı"})
        else:
            print("[DEBUG] Kamera veya model başlatılamadı")
            return jsonify(
                {"status": "error", "message": "Kamera veya model başlatılamadı"}
            )
    else:
        return jsonify({"status": "error", "message": "Tespit zaten çalışıyor"})


@bp.route("/stop_detection", methods=["POST"])
def stop_detection():
    global is_processing, picam2

    if is_processing:
        is_processing = False
        cleanup_camera()
        return jsonify({"status": "success", "message": "Tespit durduruldu"})
    else:
        return jsonify({"status": "error", "message": "Tespit zaten durdurulmuş"})
