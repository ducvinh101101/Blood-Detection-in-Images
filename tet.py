import cv2
import os

image_dir = "datasetyolo/train/images/"
label_dir = "datasetyolo/train/labels/"
output_dir = "datay/blood/"

PADDING = 20
os.makedirs(output_dir, exist_ok=True)

for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        image_file = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_file)

        if not os.path.exists(image_path):
            print(f"Không tìm thấy file ảnh: {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Không thể đọc file ảnh: {image_path}")
            continue

        h, w = img.shape[:2]
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, "r") as f:
            for idx, line in enumerate(f):
                values = line.strip().split()
                if len(values) != 5:  # Kiểm tra nếu không đúng 5 giá trị
                    print(f"Lỗi định dạng trong {label_file}, dòng {idx + 1}: {line.strip()} (yêu cầu 5 giá trị)")
                    continue  # Bỏ qua dòng lỗi, tiếp tục với dòng tiếp theo

                class_id, x_center, y_center, width, height = map(float, values)

                x_min = int((x_center - width / 2) * w)
                x_max = int((x_center + width / 2) * w)
                y_min = int((y_center - height / 2) * h)
                y_max = int((y_center + height / 2) * h)

                x_min = max(0, x_min - PADDING)
                x_max = min(w, x_max + PADDING)
                y_min = max(0, y_min - PADDING)
                y_max = min(h, y_max + PADDING)

                if x_max <= x_min or y_max <= y_min:
                    print(f"Vùng cắt không hợp lệ trong {label_file} tại dòng {idx + 1}")
                    continue

                crop = img[y_min:y_max, x_min:x_max]
                output_filename = f"blood_{os.path.splitext(label_file)[0]}_{idx}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, crop)

print("Đã hoàn thành xử lý tất cả các file!")