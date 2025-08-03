from ultralytics import YOLO

# Load the base YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' = nano version (fast and lightweight)

# Train the model
model.train(
    data="dataset/data.yaml",  # Path to your dataset config
    epochs=50,
    imgsz=640
)