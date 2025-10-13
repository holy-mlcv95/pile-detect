import cv2
import numpy as np
import os

def detect_pile(image_path, out_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=120,
    )

    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
            # Draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 3)

    # Save or display the result
    cv2.imwrite(out_path, img)
    print(f"Saved result to {out_path}")

if __name__ == "__main__":
    data_folder = "data"
    images = ["Pile1.png", "Pile2.png"]
    for im_name in images:
        im_path = os.path.join(data_folder, im_name)
        out_path = os.path.join(data_folder, f"result_{im_name}")
        detect_pile(im_path, out_path)