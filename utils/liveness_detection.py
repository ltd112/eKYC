import random
import torch
from utils.functions import extract_face
from facenet_pytorch import MTCNN
from liveness_detection.face_orientation import FaceOrientationDetector
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
face_orientation_detector = FaceOrientationDetector()

challenge_list = ["front", "left", "right"]

def get_question(challenge):
    """
    Generate the question or instruction based on the challenge.
    """
    if challenge == 'front':
        return "Please face forward"
    elif challenge == 'left':
        return "Please turn your face to the left"
    elif challenge == 'right':
        return "Please turn your face to the right"


def random_challenge():
    """
    Randomly select the next challenge (front, left, or right).
    """
    return random.choice(challenge_list)


def result_challenge_response(frame: np.ndarray, challenge: str, model: list, mtcnn: MTCNN):
    """
    Process the response to a challenge based on the input frame.
    """
    face, box, landmarks = extract_face(frame, mtcnn, padding=10)
    if box is not None:
        orientation = model[0].detect(landmarks)  # Using face_orientation_detector
        return orientation == challenge
    return False


def update_challenge(current_challenge: str):
    """
    Update the challenge to the next one in sequence (front -> left -> right).
    """
    current_challenge_index = challenge_list.index(current_challenge)
    if current_challenge_index + 1 < len(challenge_list):
        return challenge_list[current_challenge_index + 1]
    else:
        return "completed"  # End of challenge
