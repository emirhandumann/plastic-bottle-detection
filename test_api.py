import requests
import json
import cv2
import numpy as np
from PIL import Image
import io


def test_api(image_path):
    # API endpoint
    url = "http://localhost:5000/predict"

    try:
        # Image
        with open(image_path, "rb") as img:
            files = {"image": img}

            # Request
            response = requests.post(url, files=files)

            # Results
            result = response.json()

            if result["success"]:
                print("\nDetection Results:")
                print(json.dumps(result, indent=2, ensure_ascii=False))

                # Load image and show detections
                image = cv2.imread(image_path)
                for detection in result["detections"]:
                    bbox = detection["bbox"]
                    conf = detection["confidence"]
                    weight = detection["estimated_weight"]

                    # Bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Confidence score and weight
                    label = f"Plastic Bottle: {conf:.2f} - Weight: {weight}g"
                    cv2.putText(
                        image,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # Show result
                cv2.imshow("Detection Results", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            else:
                print(f"Error: {result['error']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":

    test_image_path = "testing/test.jpg"
    test_api(test_image_path)
