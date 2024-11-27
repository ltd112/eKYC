import os
import cv2 as cv
import numpy as np
import torch
from mtcnn import MTCNN
from deepface import DeepFace
from utils.functions import *
from utils.distance import *


# Function to verify two images using DeepFace.verify()
def verify(img1: np.ndarray, img2: np.ndarray, detector_model: MTCNN):
    # Step 1: Extract faces using MTCNN
    face1, _, _ = extract_face(img1, detector_model)
    face2, _, _ = extract_face(img2, detector_model)

    # Step 2: Save the cropped faces to the 'crop' folder
    save_path1 = "crop/face1.jpg"
    save_path2 = "crop/face2.jpg"
    save_cropped_face(face1, save_path1)
    save_cropped_face(face2, save_path2)
    
    # Step 3: Use DeepFace to verify the faces
    result = DeepFace.verify(
        img1_path=save_path1,
        img2_path=save_path2,
        model_name="Facenet512",  # Using Facenet512
        enforce_detection=False,  # Since we're using MTCNN, no need to enforce detection here
        distance_metric="cosine"  # Or other metrics like "euclidean", "L1"
    )
    
    return result

def verify_with_camera(img1: np.ndarray, detector_model: MTCNN):
    cap = cv.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam.")
            break

        # Convert the frame to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Lật gương ảnh (nếu cần thiết)
        frame_flipped = flip_image(frame_rgb)

        # Tiền xử lý ảnh
        #frame_processed = preprocess_image(frame_flipped)

        # Extract face from the webcam frame
        face_cam, _, _ = extract_face(frame_flipped, detector_model)

        result_text = "No face detected"  # Default message

        if face_cam is not None:
            # Save the webcam face to 'crop' folder
            save_path_cam = "crop/face_cam.jpg"
            save_cropped_face(face_cam, save_path_cam)

            # Verify the first image against the webcam face
            result = DeepFace.verify(
                img1_path="crop/face1.jpg",  # Pre-cropped first image
                img2_path=save_path_cam,
                model_name="Facenet512",
                enforce_detection=False,
                distance_metric="cosine"
            )

            # Process verification result
            if result['verified']:
                result_text = "Matched"
                color = (0, 255, 0)  # Green for matched
            else:
                result_text = "Not Matched"
                color = (0, 0, 255)  # Red for not matched
        else:
            color = (0, 0, 255)  # Red for no face

        #convert the frame to BGR
        frame_flipped_bgr = cv.cvtColor(frame_flipped, cv.COLOR_RGB2BGR)

        # Add the result text to the frame
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame_flipped_bgr, result_text, (50, 50), font, 1, color, 2, cv.LINE_AA)

        # Display the webcam feed with the result
        cv.imshow("Webcam", frame_flipped_bgr)

        # Exit the loop when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# Main function for testing
if __name__ == "__main__":
    # Loading images
    filename1 = "./ocr/crop_images/cropped_cccd.jpg"
    filename2 = "./file_test/datv2.jpg"
    image1 = get_image(filename1)
    image2 = get_image(filename2)

    # Device setup for MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector_model = MTCNN(device=device)

    # Step 1: Verify two images
    results = verify(image1, image2, detector_model)
    print("Verification results:", results)

    # # Step 2: Real-time verification with the camera
    # face1, _, _ = extract_face(image1, detector_model)
    # if face1 is not None:
    #     save_cropped_face(face1, "crop/face1.jpg")
    #     verify_with_camera(image1, detector_model)
    # else:
    #     print("Error: No face detected in the first image.")
