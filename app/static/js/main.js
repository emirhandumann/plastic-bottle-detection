document.addEventListener('alpine:init', () => {
    Alpine.data('bottleDetector', () => ({
        imagePreview: null,
        processing: false,
        totalPoints: 0,
        qrCode: null,

        async captureImage() {
            try {
                this.processing = true;
                
                // Capture image from camera
                const response = await fetch('/api/capture');
                const data = await response.json();
                
                if (data.success) {
                    this.imagePreview = `data:image/jpeg;base64,${data.image}`;
                    
                    // Process image and detect bottles
                    const detectResponse = await fetch('/api/detect', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            image: data.image
                        })
                    });
                    
                    const detectData = await detectResponse.json();
                    
                    if (detectData.success) {
                        this.totalPoints = detectData.detections.reduce((sum, d) => sum + d.points, 0);
                        this.qrCode = detectData.qr_code;
                    } else {
                        console.error('Tespit hatası:', detectData.error);
                        alert('Şişe tespiti sırasında bir hata oluştu.');
                    }
                } else {
                    console.error('Kamera hatası:', data.error);
                    alert('Kamera görüntüsü alınamadı.');
                }
            } catch (error) {
                console.error('İşlem hatası:', error);
                alert('Bir hata oluştu. Lütfen tekrar deneyin.');
            } finally {
                this.processing = false;
            }
        }
    }));
}); 