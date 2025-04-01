import cv2
import numpy as np

def preprocess_blood_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
    image = cv2.resize(image, (224, 224))  # Resize về 224x224
    image = image.astype(np.float32) / 255.0  # Chuẩn hóa về [0,1]

    hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])  # Cân bằng histogram

    final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return final_image

image_path = "Data/noblood/240_F_142771967_yYP3UoaeIYT3XjV3YfvxRa1JdVGHBqJz.jpg"  # Thay đường dẫn ảnh
image = cv2.imread(image_path)
import matplotlib.pyplot as plt

if image is not None:
    processed_image = preprocess_blood_image(image)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(processed_image)
    axes[1].set_title("Processed Image")
    axes[1].axis("off")

    plt.show()

