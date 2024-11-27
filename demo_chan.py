import random
import cv2 as cv
import numpy as np
import dlib
import torch
import os
from liveness_detection.faceOrientationDetector import FaceOrientationDetector
from utils.functions import extract_face

# Đường dẫn đến tệp shape_predictor_68_face_landmarks.dat
landmark_path = os.path.join('models/shape_predictor_68_face_landmarks.dat')

# Tải model dlib cho nhận diện khuôn mặt và landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_path)

def random_challenge():
    return random.choice(['right', 'left', 'front'])

def get_question(challenge):
    """
    Generate a question or instruction based on the challenge.
    """
    if challenge in ['right', 'left', 'front']:
        return "Please turn your face to the {}".format(challenge)

def get_challenge_and_question():
    challenge = random_challenge()
    question = get_question(challenge)
    return challenge, question

def face_response(challenge: str, landmarks: np.ndarray, model: FaceOrientationDetector):
    """
    Check if the user's face orientation matches the challenge.
    """
    orientation = model.detect(landmarks)
    print(f"Detected Orientation: {orientation}, Challenge: {challenge}")
    return orientation == challenge

def result_challenge_response(frame: np.ndarray, challenge: str, model: FaceOrientationDetector, detector, predictor):
    """
    Process the response to a challenge based on the input frame using dlib.
    """
    # Chuyển đổi ảnh sang định dạng RGB
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = detector(gray)
    
    if len(faces) > 0:
        # Lấy landmark cho khuôn mặt đầu tiên
        face = faces[0]
        landmarks = predictor(gray, face)

        # Chuyển đổi landmarks thành mảng numpy
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Kiểm tra kết quả
        is_correct = face_response(challenge, landmarks_points, model)
        return is_correct
    return False


if __name__ == '__main__':
    video = cv.VideoCapture(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    face_orientation_detector = FaceOrientationDetector()

    model = face_orientation_detector
    challenge, question = get_challenge_and_question()
    challenge_is_correct = False

    count = 0
    while True:
        ret, frame = video.read()

        if ret:
            frame = cv.flip(frame, 1)

            if not challenge_is_correct:
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                challenge_is_correct = result_challenge_response(rgb_frame, challenge, model, detector, predictor)

                cv.putText(
                    frame,
                    f"Question: {question}",
                    (20, 20),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv.imshow("Challenge", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            count += 1

            if challenge_is_correct and count >= 100:
                challenge, question = get_challenge_and_question()
                print(question)
                challenge_is_correct = False
                count = 0
        else:
            break
