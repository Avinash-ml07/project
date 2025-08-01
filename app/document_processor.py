import pytesseract 
import cv2
from PIL import Image
import boto3
from textractor import Textractor

class DocumentProcessor:
    def __init__(self):
        self.textract_client = Textractor(profile_name="default")

    def process_with_tesseract(self, image_path):
        img = cv2.imread(image_path)  # Read the image using OpenCV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        denoised = cv2.medianBlur(gray, 5)  # Denoise the image
        custom_config = r'--oem 3 --psm 6 -l eng'
        text = pytesseract.image_to_string(denoised, config=custom_config)
        return text
    def process_with_textract(self, document_path):
        document = self.textract_client.detect_document_text(document_path)
        return document.text