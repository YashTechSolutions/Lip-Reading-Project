from ultralytics import YOLO
import cv2

model = YOLO("./lip_yolo_run/detect/train/weights/best.pt")
results = model.predict('./test_data',save=True)

