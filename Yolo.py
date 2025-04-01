from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data="D:/user/Desktop/Github/Blood-Detection-in-Images/datasetyolo/data.yaml", epochs=1, batch=32, imgsz=640, save_period=1)