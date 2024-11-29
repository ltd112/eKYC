from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse , FileResponse
# import httpexception

# add middleware cors
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os

from ultralytics import YOLO
from paddleocr import PaddleOCR

import cv2 as cv
import torch
from deepface import DeepFace
from mtcnn import MTCNN
from utils.functions import *

# Import các hàm từ module
from utils.image_processing import preprocess_image
from utils.ocr_utils import extract_cccd_content
from utils.yolo_detection import detect_and_crop_cccd

# from utils.liveness_detection import get_question, result_challenge_response, update_challenge
# from liveness_detection.face_orientation import FaceOrientationDetector

from liveness_detection.faceOrientationDetector import FaceOrientationDetector
import dlib


app = FastAPI()

origins = [
    "http://localhost:3408",  #địa chỉ frontend Angular
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo model YOLO và PaddleOCR
MODEL_PATH = "./models/trained_yolov8_model.pt"
model = YOLO(MODEL_PATH)
ocr = PaddleOCR(use_angle_cls=True, lang='vi')

# Device setup cho MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector_model = MTCNN(device=device)

#mtcnn = MTCNN(device=device)
face_orientation_detector = FaceOrientationDetector()

# Đường dẫn ảnh căn cước đã cắt và ảnh mặt
cropped_cccd_path = "crop/cropped_cccd.jpg"
face1_path = "crop/face1.jpg"

# Load the face detector and landmarks predictor
landmark_path = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_path)

# Initialize face orientation detector
face_orientation_detector = FaceOrientationDetector()

#done
@app.post("/detect_orientation/")
async def detect_orientation(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(file.file.read())
        img = get_image(temp_file)
        face, _, _ = extract_face(img, detector_model)
        # Detect faces in the image
        face_path = "crop/face_liveness.jpg"
        save_cropped_face(face, face_path)
        face_liveness = get_image(face_path)
        gray = cv.cvtColor(face_liveness, cv.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        
        # Check if no face is detected
        if len(faces) == 0:
            os.remove(temp_file)
            return JSONResponse(content={"error": "Không phát hiện khuôn mặt trong ảnh."}, status_code=400)
        
        # Process the first detected face (if multiple faces are detected)
        face = faces[0]
        landmarks = predictor(gray, face)
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Detect face orientation (front, left, right)
        orientation = face_orientation_detector.detect(landmarks_points)
        os.remove(temp_file)
        return {"orientation": orientation}
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
#done            
# face verification API         
@app.post("/verify-with-camera/")
async def verify_with_camera(file: UploadFile = File(...)):
    try:
        img1 = get_image(cropped_cccd_path)  # Dùng hàm get_image
        face1, _, _ = extract_face(img1, detector_model)
        if face1 is not None:
            save_cropped_face(face1, face1_path)
        else:
            print("Error: No face detected in cropped_cccd.jpg.")
        
        # Lưu tạm thời ảnh webcam
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(file.file.read())

        # Đọc ảnh webcam bằng get_image
        img_cam = get_image(temp_file)

        # Trích xuất mặt từ ảnh webcam
        face_cam, _, _ = extract_face(img_cam, detector_model)
        if face_cam is None:
            return JSONResponse(content={"error": "Không phát hiện khuôn mặt trong ảnh từ webcam."}, status_code=400)

        # Lưu mặt từ webcam
        save_cropped_face(face_cam, "crop/face_cam.jpg")

        # So sánh mặt từ webcam với mặt đã lưu từ căn cước (face1.jpg)
        result = DeepFace.verify(
            img1_path=face1_path,  # So sánh với ảnh mặt từ căn cước (face1.jpg)
            img2_path="crop/face_cam.jpg",  # Mặt từ webcam
            model_name="Facenet512",
            enforce_detection=False,
            distance_metric="cosine"
        )

        # Xóa tệp tạm
        os.remove(temp_file)

        # Kiểm tra khoảng cách với ngưỡng
        threshold = 0.6
        verified = result["distance"] <= threshold

        # Trả về kết quả xác thực
        return {
            "verified": verified,
            "distance": result["distance"],
            "similarity_metric": "cosine",
            "threshold": threshold
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#done
# API endpoint
#OCR API
# trả về file căn cước đã cắt từ yolov8
@app.get("/get-cropped-cccd/")
async def get_cropped_cccd():
    # Đường dẫn file ảnh đã cắt
    cropped_image_path = "crop/cropped_cccd.jpg"

    # Kiểm tra nếu file không tồn tại
    if not os.path.exists(cropped_image_path):
        return JSONResponse(content={"error": "Ảnh căn cước đã cắt không tồn tại."}, status_code=404)

    # Trả về file ảnh
    return FileResponse(
        path=cropped_image_path,
        media_type="image/jpeg",
        filename="cropped_cccd.jpg"
    )
#done
@app.post("/process-cccd/")
async def process_cccd(file: UploadFile = File(...)):
    try:
        # Lưu file tạm thời
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(file.file.read())

        # Cắt và lưu vùng chứa CCCD (chỉ lấy vùng đầu tiên)
        output_dir = "crop"
        cropped_image_path = detect_and_crop_cccd(temp_file, model, output_dir=output_dir)

        if cropped_image_path is None:
            return JSONResponse(content={"error": "Không tìm thấy vùng CCCD."}, status_code=400)

        # Đọc ảnh đã cắt và xử lý OCR
        cropped_image = get_image(cropped_image_path)
        
        # Tiền xử lý ảnh
        processed_image = preprocess_image(cropped_image)
        
        # Kiểm tra xem ảnh đã được xử lý đúng hay chưa
        if processed_image is None:
            return JSONResponse(content={"error": "Lỗi khi xử lý ảnh."}, status_code=400)

        # Trích xuất thông tin từ ảnh đã xử lý
        content = extract_cccd_content(ocr  , processed_image)  # OCR và trích xuất thông tin

        # Xóa file tạm
        os.remove(temp_file)
        
        # Trả về kết quả OCR
        return {
            "results": content
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
# API để lấy một ảnh cụ thể
# Đường dẫn đến thư mục lưu ảnh
UPLOAD_FOLDER = "crop"

# Kiểm tra nếu thư mục chưa tồn tại thì tạo mới
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Đặt tên file cố định là img_cccd.jpg
        file_name = "img_cccd.jpg"
        file_path = os.path.join(UPLOAD_FOLDER, file_name)

        # Lưu ảnh vào thư mục
        with open(file_path, "wb") as image_file:
            image_file.write(await file.read())

        return JSONResponse(content={"message": f"Image '{file_name}' uploaded successfully!"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Liveness detection API
# Variables to store the current challenge and the state of the user
# current_challenge = "front"  # Initialize with the first challenge


# @app.post("/liveness-challenge/")
# async def liveness_challenge(file: UploadFile = File(...)):
#     global current_challenge

#     # Save the uploaded file temporarily
#     temp_file = f"temp_{file.filename}"
#     with open(temp_file, "wb") as f:
#         f.write(file.file.read())

#     # Read the image
#     frame = cv.imread(temp_file)

#     # Convert the image to RGB
#     rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

#     # Check if the user's face response is correct
#     is_correct = result_challenge_response(rgb_frame, current_challenge, [face_orientation_detector], detector_model)

#     # Delete temporary file
#     os.remove(temp_file)

#     # Determine if the response is correct
#     if is_correct:
#         # Move to the next challenge
#         current_challenge = update_challenge(current_challenge)
#         return {"status": "success", "message": "Correct!", "next_challenge": get_question(current_challenge)}
    
#     return JSONResponse(content={"error": "Incorrect response. Please try again."}, status_code=400)


# @app.get("/get-next-challenge/")
# async def get_next_challenge():
#     """
#     Endpoint to get the current challenge.
#     """
#     if current_challenge == "completed":
#         return {"status": "completed", "message": "You have completed all challenges."}
    
#     return {"current_challenge": get_question(current_challenge)}
    
    
    

