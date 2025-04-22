from flask import jsonify, request
from app.api import bp
import cv2
import numpy as np
import base64
import qrcode
import io
import os
from PIL import Image
import torch
import sys
import json
from picamera2 import Picamera2
import time

# CUDA kullanımını devre dışı bırak
torch.backends.cudnn.enabled = False

# Global değişkenler
picam2 = None
model = None

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
        time.sleep(1)  # Sistem kaynaklarının serbest kalması için bekle
        
        picam2 = Picamera2()
        preview_config = picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        picam2.configure(preview_config)
        
        try:
            picam2.start()
            time.sleep(2)  # Kameranın başlaması için bekle
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
    global model
    try:
        # Model yolunu düzelt
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 
                                "plastic-bottle-detection", "plastic_bottle_detection", "exp1", "weights", "best.pt")
        
        print(f"Model path: {model_path}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # YOLO model yükleme ayarları
        import torch.nn as nn
        from ultralytics.nn.tasks import DetectionModel
        torch.nn.Module.dump_patches = True
        
        # Model yükleme
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        # Model ayarları
        model.to('cpu')
        model.eval()
        
        print("Model successfully loaded")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None
        return False

# Başlangıçta kamera ve modeli yükle
initialize_camera()
load_model()

@bp.route('/capture', methods=['GET'])
def capture():
    global picam2
    
    if picam2 is None or not initialize_camera():
        return jsonify({'success': False, 'error': 'Camera initialization failed'}), 500
    
    try:
        # Görüntü yakala
        image = picam2.capture_array()
        
        # PIL Image'e çevir (zaten RGB formatında)
        pil_image = Image.fromarray(image)
        
        # Base64'e çevir
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': img_str
        })

    except Exception as e:
        print(f"Capture error: {str(e)}")
        cleanup_camera()  # Hata durumunda kamerayı temizle
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/detect', methods=['POST'])
def detect():
    global model
    
    if model is None and not load_model():
        return jsonify({'success': False, 'error': 'Model loading failed'}), 500

    try:
        # Get image from request
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        # Predict with YOLO
        with torch.no_grad():
            results = model.predict(source=image_np, conf=0.25, device='cpu')
        
        # Process results
        detections = []
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    try:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(float, xyxy)
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Calculate points based on bottle size
                        height = y2 - y1
                        points = calculate_points(height)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls,
                            'points': points
                        })
                    except Exception as box_error:
                        print(f"Error processing box: {str(box_error)}")
                        continue

        # Generate QR code
        qr_data = generate_qr_code(sum(d['points'] for d in detections))
        
        return jsonify({
            'success': True,
            'detections': detections,
            'qr_code': qr_data
        })

    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

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