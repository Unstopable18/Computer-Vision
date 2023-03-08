from ultralytics import YOLO

model=YOLO('D:\Python\Computer-Vision\yolo\Yolo-weights\yolov8n.pt')
results = model.train(data="D:\Python\Computer-Vision\yolo\ppe-detection\Dataset\data.yaml", epochs=3, plots=True)
print(results)