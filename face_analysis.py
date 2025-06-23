import os
from deepface import DeepFace
import cv2
import numpy as np

def analyze_face():
    """
    Analyzes an input image for face recognition and facial attributes.
    
    Returns:
        dict: Dictionary containing gender and emotion. Empty strings for undetected attributes.
    """
    input_image_path = 'images/user.jpg'
    reference_folder='images/references'

    result = {
        'gender': '',
        'emotion': ''
    }
    
    try:
        # Step 1: Read the input image
        input_img = cv2.imread(input_image_path)
        if input_img is None:
            print(f"Error: Could not load image at {input_image_path}")
            return result

        # Step 2: Face detection and attribute analysis
        analysis = DeepFace.analyze(
            img_path=input_image_path,
            actions=['gender', 'emotion'],
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

        # result['age'] = str(analysis.get('age', '')) if 'age' in analysis else ''
        result['gender'] = analysis.get('dominant_gender', '') if 'dominant_gender' in analysis else ''
        result['emotion'] = analysis.get('dominant_emotion', '') if 'dominant_emotion' in analysis else ''

        # Step 3: Face recognition against reference images
        if os.path.exists(reference_folder):
            for ref_image in os.listdir(reference_folder):
                if ref_image.lower().endswith(('.jpg', '.jpeg', '.png')):
                    ref_path = os.path.join(reference_folder, ref_image)
                    try:
                        # Verify if the input image matches the reference image
                        verification = DeepFace.verify(
                            img1_path=input_image_path,
                            img2_path=ref_path,
                            model_name='OpenFace',
                            detector_backend='opencv',
                            enforce_detection=False,
                            silent=True
                        )
                        if verification['verified']:
                            # Extract name from filename (remove extension)
                            result['name'] = os.path.splitext(ref_image)[0]
                            break
                    except Exception as e:
                        print(f"Error comparing with {ref_image}: {str(e)}")
        else:
            print(f"Reference folder {reference_folder} does not exist.")

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return result

    return result

if __name__ == "__main__":
    # # Path to the input image
    # input_image = "user.jpg"
    # # Path to the folder containing reference images (named as 'person_name.jpg')
    # reference_folder = "references"
    # Analyze the image
    result = analyze_face()
    
    # Print the results
    print("Analysis Result:")
    print(f"Gender: {result['gender']}")
    print(f"Emotion: {result['emotion']}")
