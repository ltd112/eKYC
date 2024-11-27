import numpy as np

class FaceOrientationDetector():
    """This class detects the orientation of a face in an image."""
    def __init__(self):
        # Frontal angle range
        self.frontal_range = [25, 60]
        # Threshold for tilt detection
        self.tilt_threshold = 15

    def calculate_angle(self, v1, v2):
        '''
        Calculate the angle between 2 vectors v1 and v2
        '''
        v1 = np.array(v1)
        v2 = np.array(v2)
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine = np.clip(cosine, -1.0, 1.0)  # Clip to avoid numerical issues
        rad = np.arccos(cosine)
        degrees = np.degrees(rad)
        return np.round(degrees)

    def detect(self, landmarks: np.ndarray):
        """
        Detect the orientation of a face based on landmarks.

        Parameters:
            landmarks (np.ndarray): A list of points representing the positions on the face.

        Returns:
            str: The face orientation ('front', 'left', 'right', 'up', 'down', 'tilted left', 'tilted right').
        """
        # Trích xuất các điểm chính từ landmarks
        left_eye = np.mean(landmarks[36:42], axis=0)  # Trung bình các điểm mắt trái
        right_eye = np.mean(landmarks[42:48], axis=0)  # Trung bình các điểm mắt phải
        nose_tip = landmarks[30]  # Chóp mũi
        mouth_center = np.mean(landmarks[48:68], axis=0)  # Trung tâm miệng
        chin = landmarks[8]  # Điểm cằm

        # Vector định hướng
        eye_vector = right_eye - left_eye
        eye_to_nose_vector = nose_tip - ((left_eye + right_eye) / 2)
        chin_vector = chin - nose_tip

        # Tính các góc
        left_angle = self.calculate_angle(eye_vector, nose_tip - left_eye)
        right_angle = self.calculate_angle(-eye_vector, nose_tip - right_eye)
        vertical_angle = self.calculate_angle(eye_to_nose_vector, [0, 1])  # Vector dọc
        chin_angle = self.calculate_angle(chin_vector, [0, 1])

        # Ghi nhật ký các góc
        print(f"Angles - Left: {left_angle}, Right: {right_angle}, Vertical: {vertical_angle}, Chin: {chin_angle}")

        # Xác định hướng khuôn mặt
        if self.frontal_range[0] <= left_angle <= self.frontal_range[1] \
                and self.frontal_range[0] <= right_angle <= self.frontal_range[1]:
            if vertical_angle < self.tilt_threshold:
                return 'front'
            elif chin_angle > 20:
                return 'down'
            elif chin_angle < -20:
                return 'up'

        elif left_angle < right_angle:
            return 'right'

        elif right_angle < left_angle:
            return 'left'

        # Thêm hướng nghiêng
        if chin_angle > self.tilt_threshold:
            return 'tilted left'
        elif chin_angle < -self.tilt_threshold:
            return 'tilted right'

        return 'unknown'

