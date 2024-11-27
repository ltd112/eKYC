import cv2
import torch
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
import numpy as np

# 1. Dùng YOLOv8 để cắt đối tượng CCCD
def detect_and_crop_cccd(image_path, model_path):
    """
    image_path: Đường dẫn đến ảnh CCCD
    model_path: Đường dẫn đến model YOLOv8 đã huấn luyện
    """
    # Load mô hình YOLOv8 đã train
    model = YOLO(model_path)
    # Dự đoán
    results = model(image_path)  # Kết quả trả về là một list
    # Lấy tọa độ bounding box từ kết quả trả về
    detections = results[0].boxes.xyxy.cpu().numpy()  # Truy cập kết quả dự đoán của ảnh đầu tiên (results[0])
    cropped_images = []
    image = cv2.imread(image_path)
    for det in detections:
        if len(det) >= 4:  # Kiểm tra đủ tọa độ bounding box
            x1, y1, x2, y2 = map(int, det[:4])  # Lấy tọa độ bounding box
            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)
    return cropped_images


# 2. Xử lý ảnh CCCD
def preprocess_image(image):
    """
    Hàm tiền xử lý ảnh cải tiến:
    - Chuyển ảnh sang xám.
    - Khử nhiễu với thông số tối ưu.
    - Tăng độ tương phản bằng histogram equalization.
    - Làm nét ảnh bằng Laplacian.
    """
    # Chuyển ảnh sang màu xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Khử nhiễu bằng GaussianBlur
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Khử nhiễu thêm bằng fastNlMeansDenoising
    denoised = cv2.fastNlMeansDenoising(blurred, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Tăng cường độ tương phản bằng Histogram Equalization
    enhanced = cv2.equalizeHist(denoised)

    # Làm nét ảnh bằng Laplacian
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(enhanced - 0.5 * laplacian)

    return sharpened

# Danh sách các cụm từ không mong muốn
unwanted_phrases = [
    "CONG HOA XA HOI CHU NGHIA VIET NAM", 
    "SOCIALIST REPUBLIC OF VIET NAM", 
    "CAN CUOC CONG DAN",
]

# 4. Đọc nội dung từ ảnh CCD bằng PaddleOCR và trích xuất thông tin bằng regex
def extract_cccd_content(image):
    """
    image: Ảnh đã được cắt
    """
    # Khởi tạo PaddleOCR với ngôn ngữ tiếng Việt
    ocr = PaddleOCR(
        use_angle_cls=True, 
        lang='vi', 
        # use_gpu=True, 
        # det_db_box_thresh=0.3, 
        # det_db_unclip_ratio=1.5,
        det_algorithm='DB',
        rec_algorithm='CRNN'
    )
    
    # Chạy OCR trên ảnh
    result = ocr.ocr(image, cls=True)
    
    # In ra toàn bộ kết quả OCR để xem dữ liệu nhận diện được trước khi lọc
    print("Kết quả OCR:", result)

    # Kết quả OCR dạng text (chỉ văn bản từ kết quả OCR)
    detected_texts = [clean_text(line[-1][0]) for line in result[0]]

    # Biểu thức chính quy (regex) để tìm các thông tin cần thiết
    cccd_regex = r"\b\d{9,12}\b"  # Tìm số căn cước công dân (9-12 chữ số)
    name_regex = r"(Ho va ten|Full name)\s+([A-Z]{2,25}(?: [A-Z]{2,25}){0,4})" # Regex tìm kiếm "Full name" và lấy tên sau đó
    dob_regex = r"\d{2}[\/\-]\d{2}[\/\-]\d{4}"  # Nhận diện ngày sinh với định dạng phổ biến

    cccd = None
    name = None
    dob = None

    # Áp dụng regex để trích xuất thông tin từ các dòng text
    combined_text = " ".join(detected_texts)
    
    # Tìm căn cước công dân
    match_cccd = re.search(cccd_regex, combined_text)
    if match_cccd:
        cccd = match_cccd.group(0)

    # Tìm họ và tên
    match_name = re.search(name_regex, combined_text)
    if match_name:
        name = match_name.group(2)
    else:
    # Nếu không tìm thấy, sử dụng regex dự phòng
        name_regex_fallback = r"\b[A-Z]{2,25}(?: [A-Z]{2,25})*\b"
        match_name = re.search(name_regex_fallback, combined_text)
        if match_name:
            # Nếu regex dự phòng tìm thấy, kiểm tra lại với danh sách cụm từ không mong muốn
            extracted_name = match_name.group()
            if any(unwanted_phrase in extracted_name for unwanted_phrase in unwanted_phrases):
                name = None  # Bỏ qua nếu tên không hợp lệ
            else:
                name = extracted_name


    # Tìm ngày sinh
    match_dob = re.search(dob_regex, combined_text)
    if match_dob:
        dob = match_dob.group(0)

    # Trả về kết quả là một dictionary chứa các thông tin cần thiết
    return {
        # "Dữ liệu OCR": detected_texts,
        "Số CCCD": cccd,
        "Họ và tên": name,
        "Ngày sinh": dob
    }


# 5. Làm sạch văn bản OCR
def clean_text(text):
    """
    Làm sạch và tách riêng số CCCD từ chuỗi.
    """
    # Loại bỏ các cụm từ không mong muốn
    for unwanted in unwanted_phrases:
        text = text.replace(unwanted, "")
    # Biểu thức chính quy tìm số trong chuỗi (dành cho các chuỗi như 's6/No051202004380')
    match = re.search(r"\d{9,12}$", text)
    if match:
        return match.group(0)  # Trả về số CCCD nếu tìm thấy
    
    # Xử lý thông thường nếu không tìm thấy số
    text = re.sub(r"[^A-Za-z0-9\s\/\-]", "", text)  # Loại bỏ ký tự không hợp lệ
    text = re.sub(r"\s+", " ", text).strip()  # Xóa khoảng trắng thừa
    return text

# Hàm chính
if __name__ == "__main__":
    image_path = "D:/Nam_5/KLTN/e-kyc/file_test/ccDat.jpg"  # Đường dẫn ảnh gốc
    model_path = "trained_yolov8_model.pt"  # Đường dẫn model YOLOv8 đã train

    # Tạo thư mục crop_images nếu chưa có
    crop_images_dir = "./crop_images"
    if not os.path.exists(crop_images_dir):
        os.makedirs(crop_images_dir)

    # Bước 1: Dùng YOLOv8 để cắt vùng chứa CCD
    cropped_images = detect_and_crop_cccd(image_path, model_path)
    for idx, cropped_image in enumerate(cropped_images):
        cropped_image_path = os.path.join(crop_images_dir, f"cropped_cccd.jpg")
        cv2.imwrite(cropped_image_path, cropped_image)  # Lưu ảnh đã cắt vào thư mục crop_images

        # Bước 2: Xử lý ảnh (khử nhiễu và làm nét ảnh)
        processed_image = preprocess_image(cropped_image)
        processed_image_path = os.path.join(crop_images_dir, f"processed_cropped_cccd.jpg")
        cv2.imwrite(processed_image_path, processed_image)  # Lưu ảnh đã xử lý vào thư mục crop_images

        # Bước 3: OCR và trích xuất nội dung bằng regex
        content = extract_cccd_content(processed_image)
        print(f"Nội dung từ ảnh {processed_image_path}:", content)
