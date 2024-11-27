import random
import numpy as np
import cv2 as cv
from liveness_detection.faceOrientationDetector import FaceOrientationDetector

def random_challenge():
    return random.choice(['right', 'left', 'front'])

def get_question(challenge):
    if challenge in ['right', 'left', 'front']:
        return f"Please turn your face to the {challenge}"

def get_challenge_and_question():
    challenge = random_challenge()
    question = get_question(challenge)
    return challenge, question

def face_response(challenge: str, landmarks: np.ndarray, model: FaceOrientationDetector):
    orientation = model.detect(landmarks)
    return orientation == challenge

def result_challenge_response(frame: np.ndarray, challenge: str, model: FaceOrientationDetector, detector, predictor):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])
        is_correct = face_response(challenge, landmarks_points, model)
        return is_correct
    return False
