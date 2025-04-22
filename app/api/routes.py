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
import sys

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
    # Güvenli mod için özel ayarlar
    model = YOLO(model_path)
    model.model.eval()  # Değerlendirme moduna al
    model.model.float()  # Float32 kullan
    model.model.cpu()   # CPU'ya taşı
    
    # Güvenli yükleme için ek ayarlar
    torch.set_grad_enabled(False)  # Gradyanları devre dışı bırak
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
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

        # Predict with YOLO (güvenli mod ayarları)
        with torch.no_grad():
            results = model.predict(
                source=image_np,
                device='cpu',
                half=False,
                imgsz=640,
                conf=0.25,
                save=False
            )

        # Process results
        detections = []
        if len(results) > 0:
            result = results[0]  # İlk sonucu al
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    try:
                        # Numpy array'e dönüştür
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