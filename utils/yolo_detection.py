import cv2
import os
from ultralytics import YOLO

def detect_and_crop_cccd(image_path: str, model: YOLO, output_dir: str = "crop"):
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Đọc ảnh và chạy mô hình
    image = cv2.imread(image_path)
    results = model(image_path)
    detections = results[0].boxes.xyxy.cpu().numpy()

    if detections.size == 0:  # Không phát hiện được vùng
        return None

    # Lấy vùng phát hiện đầu tiên
    x1, y1, x2, y2 = map(int, detections[0][:4])  # Chỉ lấy tọa độ của vùng đầu tiên
    cropped = image[y1:y2, x1:x2]

    # Lưu vùng ảnh đã cắt vào thư mục
    cropped_path = os.path.join(output_dir, "cropped_cccd.jpg")
    cv2.imwrite(cropped_path, cropped)

    return cropped_path  # Trả về đường dẫn ảnh đã cắt