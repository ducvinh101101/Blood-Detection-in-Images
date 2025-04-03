import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Tải mô hình một lần duy nhất
yolo_model = YOLO("my_trained_model.pt")
vgg_model = load_model("hyper_blood_3class.h5")


def detect_blood_yolo(img):
    # Chuyển đổi ảnh sang RGB một lần
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Dự đoán với YOLO
    results = yolo_model.predict(img_rgb, conf=0.2, verbose=False)  # Tắt log để nhanh hơn

    boxes = []
    labels = []
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)

    return boxes, labels


def draw_yolo_boxes(img, boxes, labels):
    img_with_boxes = img.copy()
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        color = (0, 255, 0) if label in ['blood'] else (255, 0, 0)
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img_with_boxes, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img_with_boxes


def classify_blood_vgg16(img, box):
    x_min, y_min, x_max, y_max = box
    roi = img[y_min:y_max, x_min:x_max]
    roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)  # Dùng INTER_AREA để nhanh hơn
    roi_input = np.expand_dims(roi_resized / 255.0, axis=0)  # Chuẩn hóa inline
    prediction = vgg_model.predict(roi_input, verbose=0)  # Tắt log
    label = 'blood' if prediction[0][0] > 0.5 else 'noblood'
    return label, prediction[0][0]


def process_blood_region(img, box):
    x_min, y_min, x_max, y_max = box
    roi = img[y_min:y_max, x_min:x_max]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
    img[y_min:y_max, x_min:x_max] = gray_roi
    return img


def process_image(image_path):
    # Đọc ảnh một lần
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể đọc ảnh từ đường dẫn cung cấp.")

    # Phát hiện với YOLO
    boxes, yolo_labels = detect_blood_yolo(img)

    # Vẽ bounding box
    img_with_boxes = draw_yolo_boxes(img, boxes, yolo_labels)
    cv2.imwrite('yolo_output.jpg', img_with_boxes)  # Lưu thay vì hiển thị

    # Xử lý từng vùng
    for i, (box, yolo_label) in enumerate(zip(boxes, yolo_labels)):
        print(f"Vùng {i + 1}:")
        print(f"  - YOLO: {yolo_label} (Bounding box: {box})")

        if yolo_label in ['blood', 'flood']:
            vgg_label, vgg_confidence = classify_blood_vgg16(img, box)
            print(f"  - VGG16: {vgg_label} (Confidence: {vgg_confidence:.4f})")

            if vgg_label == 'blood' and vgg_confidence > 0.7:
                img = process_blood_region(img, box)
                print("  - Đã chuyển vùng thành màu xám")
        else:
            print("  - Không xử lý (không phải blood/flood theo YOLO)")
        print("-" * 50)

    # Lưu ảnh kết quả
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, img)
    return output_path


if __name__ == "__main__":
    image_path = "Data/blood/0_w_blood_37_jpg.rf.9b8febad89a9711bf559cec5b58e0ed7.jpg"  # Đường dẫn đến ảnh đầu vào
    result = process_image(image_path)
    print(f"Ảnh kết quả được lưu tại: {result}")