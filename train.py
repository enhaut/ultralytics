from ultralytics import YOLO

# Load a model
model = YOLO('yolov8.yaml')

results = model.train(data='data/data.yaml', epochs=10, imgsz=640, device='mps', workers=0)
