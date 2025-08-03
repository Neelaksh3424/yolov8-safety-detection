from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")  # Update path if needed

# Run detection on an image
results = model("test.jpg", show=True)  # Replace with your image name