import { useState } from "react";
import { CameraView } from "../components/CameraView";
import { DetectionResults } from "../components/DetectionResults";
import { QRCodeDisplay } from "../components/QRCodeDisplay";
import { Header } from "../components/Header";

export default function Home() {
  const [detectionResults, setDetectionResults] = useState(null);
  const [qrCodeData, setQrCodeData] = useState(null);

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <CameraView
            onDetection={setDetectionResults}
            onQRCodeGenerated={setQrCodeData}
          />
          <div className="space-y-8">
            <DetectionResults results={detectionResults} />
            <QRCodeDisplay data={qrCodeData} />
          </div>
        </div>
      </main>
    </div>
  );
}
