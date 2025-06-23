import os
from deepface import DeepFace
import cv2
import numpy as np

def analyze_face():
    """
    Analyzes an input image for face recognition and facial attributes.
    
    Returns:
        str: Emotion of user
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


    # analyze_face
    input_image_path = full_path
    result = ''
    
    try:
        # Step 1: Read the input image
        input_img = cv2.imread(input_image_path)
        if input_img is None:
            print(f"Error: Could not load image at {input_image_path}")
            return result

        # Step 2: Face detection and attribute analysis
        analysis = DeepFace.analyze(
            img_path=input_image_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )

        # If no face is detected, return empty result
        if not analysis or isinstance(analysis, dict) and 'dominant_emotion' not in analysis:
            print("No face detected in the input image.")
            return result

        # Extract attributes from analysis
        if isinstance(analysis, list):
            analysis = analysis[0]  # Take the first detected face

        result = analysis.get('dominant_emotion', '') if 'dominant_emotion' in analysis else ''


    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return result

    return f"emotion: {result}"

if __name__ == "__main__":
    # # Path to the input image
    # input_image = "user.jpg"
    # # Path to the folder containing reference images (named as 'person_name.jpg')
    # reference_folder = "references"
    # Analyze the image
    result = analyze_face()
    
    print(result)
