from ultralytics import YOLO
import cv2

model=YOLO('D:\Python\Computer-Vision\yolo\Yolo-weights\yolov8l.pt')
results=model('D:/Python/Computer-Vision/yolo/Images/3.png',show=True)
cv2.waitKey(0)