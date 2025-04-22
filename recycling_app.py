import sys
import cv2
import numpy as np
import os

# Qt platformunu X11'e zorla
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["DISPLAY"] = ":0"

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from picamera2 import Picamera2
from libcamera import controls
from ultralytics import YOLO
import qrcode
from PIL import Image
import io


class RecyclingApp(QMainWindow):
    def __init__(self):
        try:
            super().__init__()
            self.setWindowTitle("Plastik Şişe Geri Dönüşüm Sistemi")
            self.setGeometry(100, 100, 800, 600)

            # Ana widget ve layout
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            layout = QVBoxLayout(main_widget)

            # Kamera önizleme etiketi
            self.camera_label = QLabel()
            self.camera_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.camera_label)

            # Geri dönüştür butonu
            self.recycle_button = QPushButton("Geri Dönüştür")
            self.recycle_button.clicked.connect(self.start_recycling_process)
            layout.addWidget(self.recycle_button)

            # QR kod gösterim etiketi
            self.qr_label = QLabel()
            self.qr_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.qr_label)

            # Kamera ve model başlatma
            self.setup_camera()
            self.setup_model()

            # Kamera önizleme zamanlayıcısı
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_camera_preview)
            self.timer.start(30)  # 30ms = ~33 FPS
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def setup_camera(self):
        """Kamera ayarlarını yapılandır"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(main={"size": (800, 600)})
            config["transform"] = controls.Transform(hflip=0, vflip=0)
            self.camera.configure(config)
            self.camera.set_controls({"FrameDurationLimits": (33333, 33333)})
            self.camera.start()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kamera başlatılamadı: {str(e)}")

    def setup_model(self):
        """YOLO modelini yükle"""
        try:
            self.model = YOLO("plastic_bottle_detection/exp1/weights/best.pt")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Model yüklenemedi: {str(e)}")

    def update_camera_preview(self):
        """Kamera önizlemesini güncelle"""
        try:
            frame = self.camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(
                QPixmap.fromImage(qt_image).scaled(
                    self.camera_label.size(), Qt.KeepAspectRatio
                )
            )
        except Exception as e:
            print(f"Kamera önizleme hatası: {str(e)}")

    def start_recycling_process(self):
        """Geri dönüşüm işlemini başlat"""
        try:
            # Fotoğraf çek
            frame = self.camera.capture_array()

            # YOLO ile tespit
            results = self.model(frame)

            # Tespit sonuçlarını işle
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Şişe boyutunu belirle ve puan hesapla
                    x1, y1, x2, y2 = box.xyxy[0]
                    height = y2 - y1
                    points = self.calculate_points(height)

                    # QR kod oluştur
                    self.generate_qr_code(points)

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"İşlem sırasında hata oluştu: {str(e)}")

    def calculate_points(self, height):
        """Şişe boyutuna göre puan hesapla"""
        if height < 100:
            return 10  # Küçük şişe
        elif height < 200:
            return 20  # Orta şişe
        else:
            return 30  # Büyük şişe

    def generate_qr_code(self, points):
        """Puan bilgisini içeren QR kod oluştur"""
        try:
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(f"points:{points}")
            qr.make(fit=True)

            qr_image = qr.make_image(fill_color="black", back_color="white")

            # PIL Image'i QPixmap'e dönüştür
            buffer = io.BytesIO()
            qr_image.save(buffer, format="PNG")
            qr_pixmap = QPixmap()
            qr_pixmap.loadFromData(buffer.getvalue())

            # QR kodu göster
            self.qr_label.setPixmap(
                qr_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"QR kod oluşturulamadı: {str(e)}")


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = RecyclingApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {str(e)}")
