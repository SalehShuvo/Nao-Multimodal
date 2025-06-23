import cv2
import os
from datetime import datetime

def capture_and_save_image():
    """
    Capture one frame from the camera and save it to disk.
    Returns:
        str: Full path to the saved image file.

    """
    # Ensure save directory exists
    os.makedirs('./images', exist_ok=True)

    # Open camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera #{0}")

    # Capture frame
    ret, frame = cam.read()
    cam.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture image from camera")

  
    full_path = os.path.join('./images', "user.jpg")

    # Write to disk
    success = cv2.imwrite(full_path, frame)
    if not success:
        raise RuntimeError(f"Failed to write image to {full_path}")

    return full_path

if __name__ == "__main__":
    saved_path = capture_and_save_image()
    print(f"Image saved to: {saved_path}")
