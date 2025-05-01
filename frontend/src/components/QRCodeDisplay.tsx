import { QRCodeSVG } from "qrcode.react";

interface QRCodeDisplayProps {
  data: string | null;
}

export const QRCodeDisplay = ({ data }: QRCodeDisplayProps) => {
  if (!data) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">QR Kod</h2>
        <p className="text-gray-600">Henüz QR kod oluşturulmadı</p>
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">QR Kod</h2>
      <div className="flex justify-center">
        <QRCodeSVG
          value={data}
          size={200}
          level="H"
          includeMargin={true}
          className="rounded-lg"
        />
      </div>
    </div>
  );
};
