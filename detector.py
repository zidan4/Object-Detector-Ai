# object_detector.py
from ultralytics import YOLO
import cv2

def detect_objects(image_path):
    model = YOLO("yolov5s.pt")
    results = model(image_path)
    results.show()

if __name__ == "__main__":
    detect_objects("sample.jpg")
