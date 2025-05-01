import { useState, useCallback, useRef } from "react";
import Webcam from "react-webcam";
import { detectBottles } from "../services/api";

interface CameraViewProps {
  onDetection: (detections: any) => void;
  onQRCodeGenerated: (qrData: any) => void;
}

export const CameraView = ({
  onDetection,
  onQRCodeGenerated,
}: CameraViewProps) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const webcamRef = useRef<Webcam>(null);

  const handleCapture = useCallback(async () => {
    if (!webcamRef.current) return;

    try {
      setIsProcessing(true);
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;

      const response = await detectBottles(imageSrc);
      onDetection(response.detections);
      onQRCodeGenerated(response.qrData);
    } catch (error) {
      console.error("Error processing image:", error);
    } finally {
      setIsProcessing(false);
    }
  }, [onDetection, onQRCodeGenerated]);

  return (
    <div className="relative">
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className="w-full rounded-lg shadow-lg"
        videoConstraints={{
          facingMode: "environment",
        }}
      />
      <button
        onClick={handleCapture}
        disabled={isProcessing}
        className="absolute bottom-4 right-4 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
      >
        {isProcessing ? "İşleniyor..." : "Tespit Et"}
      </button>
    </div>
  );
};
