import cv2
import numpy as np
from pathlib import Path
import os
from sklearn.metrics.pairwise import cosine_similarity

# Hàm trích xuất đặc trưng từ ảnh
def extract_features(image_path):
    # Đọc ảnh và chuyển sang grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    # Resize ảnh về kích thước cố định để đồng bộ (ví dụ: 300x300)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

    # Khởi tạo ORB detector
    orb = cv2.ORB_create(nfeatures=500)

    # Tìm keypoints và descriptors
    keypoints, descriptors = orb.detectAndCompute(img, None)

    return keypoints, descriptors

# Hàm so sánh hai ảnh dựa trên descriptors
def compare_images(des1, des2):
    if des1 is None or des2 is None:
        return 0.0

    # Sử dụng BFMatcher để tìm các cặp điểm khớp
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sắp xếp các matches theo khoảng cách (distance)
    matches = sorted(matches, key=lambda x: x.distance)

    # Tính tỷ lệ khớp (số matches tốt / tổng số descriptors nhỏ hơn)
    num_good_matches = len(matches)
    total_features = min(len(des1), len(des2))

    if total_features == 0:
        return 0.0

    similarity = num_good_matches / total_features
    return similarity

# Hàm lọc ảnh trùng lặp trong thư mục
def find_duplicate_images(folder_path, similarity_threshold=0.8):
    folder = Path(folder_path)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))  # Thêm định dạng nếu cần

    # Danh sách lưu đặc trưng của từng ảnh
    features_dict = {}
    for img_path in image_files:
        keypoints, descriptors = extract_features(img_path)
        if descriptors is not None:
            features_dict[img_path] = descriptors

    # So sánh từng cặp ảnh
    duplicates = []
    processed = set()

    for img1_path in features_dict.keys():
        if img1_path in processed:
            continue
        for img2_path in features_dict.keys():
            if img1_path == img2_path or img2_path in processed:
                continue

            similarity = compare_images(features_dict[img1_path], features_dict[img2_path])
            if similarity >= similarity_threshold:
                duplicates.append((img1_path, img2_path, similarity))
                processed.add(img2_path)

        processed.add(img1_path)

    return duplicates

# Chạy chương trình
if __name__ == "__main__":
    # Đường dẫn tới thư mục chứa ảnh
    folder_path = "Data_Blood_Nude/blood"  # Thay bằng đường dẫn thư mục của bạn

    # Ngưỡng tương đồng (0.8 nghĩa là 80% giống nhau)
    threshold = 0.7

    # Tìm ảnh trùng lặp
    duplicate_pairs = find_duplicate_images(folder_path, threshold)

    # In kết quả
    if duplicate_pairs:
        print("Các cặp ảnh trùng lặp:")
        for img1, img2, sim in duplicate_pairs:
            print(f"- {img1.name} và {img2.name}: {sim:.2f} giống nhau")
    else:
        print("Không tìm thấy ảnh trùng lặp.")

    # (Tùy chọn) Xóa hoặc di chuyển ảnh trùng lặp
    # Ví dụ: Xóa ảnh thứ 2 trong cặp trùng lặp
    for img1, img2, sim in duplicate_pairs:
        os.remove(img2)  # Cẩn thận khi dùng lệnh này!
        print(f"Đã xóa {img2.name}")