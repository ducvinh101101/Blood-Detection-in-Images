import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import matplotlib.pyplot as plt


def detect_blood_yolo(image_path, yolo_model_path):
    # Load model YOLO từ file .pt bằng Ultralytics
    model = YOLO(yolo_model_path)

    # Đọc ảnh
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Dự đoán với YOLO
    results = model.predict(img_rgb, conf=0.3)

    # Lấy các bounding box và nhãn
    boxes = []
    labels = []
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  # Lấy ID lớp
            label = model.names[class_id]  # Chuyển ID thành tên lớp
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)

    return img, boxes, labels


def draw_yolo_boxes(img, boxes, labels):
    # Tạo bản sao của ảnh để vẽ
    img_with_boxes = img.copy()

    # Vẽ từng bounding box và nhãn
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        # Vẽ hình chữ nhật
        color = (0, 255, 0) if label in ['blood'] else (255, 0, 0)  # Xanh cho blood/flood, đỏ cho khác
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)
        # Ghi nhãn
        cv2.putText(img_with_boxes, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img_with_boxes


def classify_blood_vgg16(img, box, vgg_model):
    x_min, y_min, x_max, y_max = box
    roi = img[y_min:y_max, x_min:x_max]  # Cắt vùng nghi ngờ
    roi_resized = cv2.resize(roi, (224, 224))  # Kích thước đầu vào của VGG16
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # Dự đoán với VGG16
    prediction = vgg_model.predict(roi_input)
    label = 'blood' if prediction[0][0] > 0.5 else 'noblood'
    confidence = prediction[0][0]
    return label, confidence


def process_blood_region(img, box):
    x_min, y_min, x_max, y_max = box
    roi = img[y_min:y_max, x_min:x_max]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
    img[y_min:y_max, x_min:x_max] = gray_roi

    return img


def process_image(image_path, yolo_model_path, vgg_model_path):
    vgg_model = load_model(vgg_model_path)
    img, boxes, yolo_labels = detect_blood_yolo(image_path, yolo_model_path)

    img_with_boxes = draw_yolo_boxes(img, boxes, yolo_labels)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Phân vùng từ YOLO")
    plt.axis('off')
    plt.show()

    print("Kết quả xử lý:")
    print("-" * 50)
    for i, (box, yolo_label) in enumerate(zip(boxes, yolo_labels)):
        print(f"Vùng {i + 1}:")
        print(f"  - YOLO: {yolo_label} (Bounding box: {box})")

        if yolo_label in ['blood', 'flood']:
            vgg_label, vgg_confidence = classify_blood_vgg16(img, box, vgg_model)
            print(f"  - VGG16: {vgg_label} (Confidence: {vgg_confidence:.4f})")

            if vgg_label == 'blood' and vgg_confidence > 0.8:
                img = process_blood_region(img, box)
                print("  - Đã chuyển vùng thành màu xám")
        else:
            print("  - Không xử lý (không phải blood/flood theo YOLO)")
        print("-" * 50)

    # Hiển thị ảnh kết quả sau xử lý
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, img)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Ảnh sau khi xử lý")
    plt.axis('off')
    plt.show()

    return output_path
# Sử dụng code
if __name__ == "__main__":
    image_path = "Data/blood/stock-photo-angry-adult-naked-woman-blood-skin-isolated-white-domestic-violence.jpg"  # Đường dẫn đến ảnh đầu vào
    yolo_model_path = "my_trained_model.pt"
    vgg_model_path = "hyper_blood_3class.h5"

    result = process_image(image_path, yolo_model_path, vgg_model_path)
    print(f"Ảnh kết quả được lưu tại: {result}")