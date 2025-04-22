document.addEventListener('alpine:init', () => {
    Alpine.data('bottleDetector', () => ({
        processing: false,
        qrCode: null,
        totalPoints: 0,
        imagePreview: null,

        async init() {
            // Başlangıçta bir şey yapmaya gerek yok
        },

        async captureImage() {
            if (this.processing) return;
            this.processing = true;

            try {
                // Raspberry Pi kamerasından görüntü al
                const response = await fetch('/api/capture', {
                    method: 'GET'
                });

                if (!response.ok) {
                    throw new Error('Görüntü alma hatası');
                }

                const result = await response.json();
                
                if (result.success) {
                    // Görüntüyü göster
                    this.imagePreview = 'data:image/jpeg;base64,' + result.image;
                    
                    // Görüntüyü analiz et
                    const detectResponse = await fetch('/api/detect', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            image: result.image
                        })
                    });

                    const detectResult = await detectResponse.json();

                    if (detectResult.success) {
                        this.qrCode = detectResult.qr_code;
                        this.totalPoints = detectResult.detections.reduce((sum, d) => sum + d.points, 0);
                        this.drawDetections(detectResult.detections);
                    } else {
                        throw new Error(detectResult.error || 'Tespit hatası');
                    }
                } else {
                    throw new Error(result.error || 'Görüntü alınamadı');
                }
            } catch (err) {
                console.error('İşlem hatası:', err);
                alert('Hata: ' + err.message);
            } finally {
                this.processing = false;
            }
        },

        drawDetections(detections) {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                
                // Orijinal görüntüyü çiz
                ctx.drawImage(img, 0, 0);

                // Tespitleri çiz
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

                // Sonucu göster
                this.imagePreview = canvas.toDataURL('image/jpeg');
            };
            img.src = this.imagePreview;
        }
    }));
}); 