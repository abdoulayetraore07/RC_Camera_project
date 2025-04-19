from ultralytics import YOLO

model = YOLO("yolov5n.pt")
model.export(format="ncnn")

model_tiny = YOLO("yolov5s.pt")
model_tiny.export(format="ncnn")