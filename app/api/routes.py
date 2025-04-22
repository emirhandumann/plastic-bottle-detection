from flask import jsonify, request
from app.api import bp
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import qrcode
import io
import os
from PIL import Image
import torch

# CUDA kullanımını devre dışı bırak
torch.backends.cudnn.enabled = False

# Model yolunu düzelt
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 
                         "plastic_bottle_detection", "exp1", "weights", "best.pt")

print(f"Model path: {model_path}")
print(f"Model file exists: {os.path.exists(model_path)}")

# Model yükleme parametrelerini belirt
try:
    model = YOLO(model_path, task='detect')
    model.to('cpu')  # CPU'ya zorla
    print("Model successfully loaded")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@bp.route('/detect', methods=['POST'])
def detect():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500

    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(file.stream)
        image_np = np.array(image)

        # Predict with YOLO
        results = model(image_np, device='cpu', half=False)  # CPU'da float32 olarak çalıştır

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # CPU'ya taşı
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Calculate points based on bottle size
                height = y2 - y1
                points = calculate_points(height)
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class': cls,
                    'points': points
                })

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