document.addEventListener('alpine:init', () => {
    Alpine.data('bottleDetector', () => ({
        processing: false,
        qrCode: null,
        totalPoints: 0,
        video: null,
        canvas: null,
        stream: null,

        async init() {
            this.video = document.getElementById('camera');
            this.canvas = document.getElementById('overlay');

            try {
                // Raspberry Pi kamera modülü için özel ayarlar
                const constraints = {
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        deviceId: undefined
                    }
                };

                // Mevcut kameraları listele
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                console.log('Bulunan kameralar:', videoDevices);

                if (videoDevices.length > 0) {
                    // Raspberry Pi kamera modülünü bul
                    const rpiCamera = videoDevices.find(device => 
                        device.label.toLowerCase().includes('rpi') || 
                        device.label.toLowerCase().includes('raspberry') ||
                        device.label.toLowerCase().includes('camera module')
                    );

                    if (rpiCamera) {
                        console.log('Raspberry Pi kamera modülü bulundu:', rpiCamera.label);
                        constraints.video.deviceId = { exact: rpiCamera.deviceId };
                    } else {
                        console.log('Varsayılan kamera kullanılıyor');
                        constraints.video.deviceId = { exact: videoDevices[0].deviceId };
                    }

                    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
                    this.video.srcObject = this.stream;
                    
                    // Video hazır olduğunda
                    this.video.onloadedmetadata = () => {
                        this.video.play();
                        // Canvas boyutlarını ayarla
                        this.canvas.width = this.video.videoWidth;
                        this.canvas.height = this.video.videoHeight;
                    };
                } else {
                    throw new Error('Hiç kamera bulunamadı');
                }
            } catch (err) {
                console.error('Kamera erişim hatası:', err);
                alert('Kameraya erişilemedi! Hata: ' + err.message);
            }
        },

        async captureImage() {
            if (this.processing || !this.video || !this.stream) return;
            this.processing = true;

            try {
                // Görüntüyü yakala
                const canvas = document.createElement('canvas');
                canvas.width = this.video.videoWidth;
                canvas.height = this.video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(this.video, 0, 0);

                // Base64'e çevir
                const imageData = canvas.toDataURL('image/jpeg');
                const blob = await (await fetch(imageData)).blob();
                const formData = new FormData();
                formData.append('image', blob, 'capture.jpg');

                // API'ye gönder
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    this.qrCode = result.qr_code;
                    this.totalPoints = result.detections.reduce((sum, d) => sum + d.points, 0);
                    this.drawDetections(result.detections);
                } else {
                    throw new Error(result.error || 'API hatası');
                }
            } catch (err) {
                console.error('İşlem hatası:', err);
                alert('Görüntü işleme hatası: ' + err.message);
            } finally {
                this.processing = false;
            }
        },

        drawDetections(detections) {
            if (!this.canvas) return;
            
            const ctx = this.canvas.getContext('2d');
            ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.font = '16px Arial';
            ctx.fillStyle = '#00ff00';

            detections.forEach(det => {
                const [x1, y1, x2, y2] = det.bbox;
                const width = x2 - x1;
                const height = y2 - y1;

                // Kutu çiz
                ctx.strokeRect(x1, y1, width, height);

                // Etiket çiz
                const label = `Şişe: ${Math.round(det.confidence * 100)}% - ${det.points} puan`;
                ctx.fillText(label, x1, y1 - 5);
            });
        }
    }));
}); 