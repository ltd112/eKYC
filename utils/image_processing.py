import cv2
import numpy as np


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    denoised = cv2.fastNlMeansDenoising(blurred, None, h=10, templateWindowSize=7, searchWindowSize=21)
    enhanced = cv2.equalizeHist(denoised)
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(enhanced - 0.5 * laplacian)
    return sharpened