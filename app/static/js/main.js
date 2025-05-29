document.addEventListener('alpine:init', () => {
    Alpine.data('bottleDetector', () => ({
        imagePreview: null,
        processing: false,
        qrCode: null,
        smallBottles: 0,
        mediumBottles: 0,
        largeBottles: 0,
        qrCodeIsImage: false,

        async captureImage() {
            this.processing = true;
            try {
                // Kamera görüntüsünü al
                const response = await fetch('/api/capture');
                const data = await response.json();
                
                if (data.success) {
                    this.imagePreview = 'data:image/jpeg;base64,' + data.image;
                    
                    // Görüntüyü işle ve şişeleri tespit et
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
                        // Şişe sayılarını güncelle
                        this.smallBottles = detectData.debug_info.bottle_counts.small || 0;
                        this.mediumBottles = detectData.debug_info.bottle_counts.medium || 0;
                        this.largeBottles = detectData.debug_info.bottle_counts.large || 0;
                        
                        // QR kodu veya hata mesajını göster
                        this.qrCode = detectData.qr_code;
                        this.qrCodeIsImage = this.qrCode && this.qrCode.length > 100; // base64 ise uzun olur
                    } else {
                        console.error('Detection failed:', detectData.error);
                        alert('Şişe tespiti başarısız oldu. Lütfen tekrar deneyin.');
                    }
                } else {
                    console.error('Capture failed:', data.error);
                    alert('Kamera görüntüsü alınamadı. Lütfen tekrar deneyin.');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Bir hata oluştu. Lütfen tekrar deneyin.');
            } finally {
                this.processing = false;
            }
        }
    }));
}); 