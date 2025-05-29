document.addEventListener('alpine:init', () => {
    Alpine.data('bottleDetector', () => ({
        imagePreview: null,
        processing: false,
        qrCode: null,
        smallBottles: 0,
        mediumBottles: 0,
        largeBottles: 0,
        qrCodeIsImage: false,
        greenPoints: 0,

        async captureImage() {
            this.processing = true;
            try {
                // Capture image from camera
                const response = await fetch('/api/capture');
                const data = await response.json();
                
                if (data.success) {
                    this.imagePreview = 'data:image/jpeg;base64,' + data.image;
                    
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
                        // Update bottle counts
                        this.smallBottles = detectData.debug_info.bottle_counts.small || 0;
                        this.mediumBottles = detectData.debug_info.bottle_counts.medium || 0;
                        this.largeBottles = detectData.debug_info.bottle_counts.large || 0;
                        // Update green points
                        this.greenPoints = detectData.debug_info.total_points || 0;
                        // Show QR code or error message
                        this.qrCode = detectData.qr_code;
                        // PNG base64's usually start with 'iVBOR' and are long
                        this.qrCodeIsImage = this.qrCode && this.qrCode.length > 100 && this.qrCode.startsWith('iVBOR');
                    } else {
                        this.qrCode = null;
                        this.qrCodeIsImage = false;
                    }
                } else {
                    this.qrCode = null;
                    this.qrCodeIsImage = false;
                    console.error('Capture failed:', data.error);
                    alert('Camera image capture failed. Please try again.');
                }
            } catch (error) {
                this.qrCode = null;
                this.qrCodeIsImage = false;
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            } finally {
                this.processing = false;
            }
        }
    }));
}); 