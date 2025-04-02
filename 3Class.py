import os
import cv2
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_dir = "datay"
categories = ["blood", "noblood", "normal"]

X, y = [], []

def preprocess_blood_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])
    final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return final_image

for category in categories:
    path = os.path.join(data_dir, category)
    if not os.path.exists(path):
        print(f"❌ Thư mục không tồn tại: {path}")
        continue

    label = categories.index(category)
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        if not os.path.isfile(img_path):
            print(f"⚠️ Bỏ qua file không hợp lệ: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Không thể đọc ảnh: {img_path}")
            continue

        processed_img = preprocess_blood_image(img)
        X.append(processed_img)
        y.append(label)

X = np.array(X, dtype='float32')
y = np.array(y, dtype='int32')  # Nhãn là số nguyên (0, 1, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

IMG_SIZE = 224
BATCH_SIZE = 32

# Load mô hình VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

# Xây dựng mô hình cho 3 lớp
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Thêm Dropout để giảm overfitting
x = Dense(3, activation='softmax')(x)  # 3 lớp với softmax
model = Model(inputs=base_model.input, outputs=x)

# Compile với sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Dùng cho nhãn số nguyên
              metrics=['accuracy'])

# Huấn luyện
history = model.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   epochs=20,
                   batch_size=BATCH_SIZE)

# Dự đoán
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Lấy lớp có xác suất cao nhất

# Vẽ biểu đồ accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Vẽ biểu đồ loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Lưu mô hình
model.save("hyper_blood_3class.h5")