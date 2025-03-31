from ultralytics import YOLO

# Tải mô hình
model = YOLO("yolov8n.pt")

# Phát hiện đối tượng
results = model("Data_Blood_Nude/blood1/istockphoto-152990177-612x612.jpg")

# Hiển thị kết quả
for re in results:
    re.show()
else:
    print("Không phát hiện được gì hoặc ảnh không tồn tại.")