interface Detection {
  confidence: number;
  points: number;
  box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

interface DetectionResultsProps {
  results: Detection[] | null;
}

export const DetectionResults = ({ results }: DetectionResultsProps) => {
  if (!results || results.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Tespit Sonuçları
        </h2>
        <p className="text-gray-600">Henüz tespit yapılmadı</p>
      </div>
    );
  }

  const totalPoints = results.reduce(
    (sum, detection) => sum + detection.points,
    0
  );

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">
        Tespit Sonuçları
      </h2>
      <div className="space-y-4">
        {results.map((detection, index) => (
          <div key={index} className="border-b pb-4 last:border-b-0">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Tespit #{index + 1}</span>
              <span className="text-green-600 font-semibold">
                {detection.confidence.toFixed(2)}%
              </span>
            </div>
            <div className="mt-2">
              <span className="text-gray-600">Puan:</span>
              <span className="ml-2 font-semibold text-green-600">
                {detection.points}
              </span>
            </div>
          </div>
        ))}
        <div className="pt-4 border-t">
          <span className="text-gray-600">Toplam Puan:</span>
          <span className="ml-2 text-2xl font-bold text-green-600">
            {totalPoints}
          </span>
        </div>
      </div>
    </div>
  );
};
