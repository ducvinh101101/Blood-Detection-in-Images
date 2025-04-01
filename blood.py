import os
import cv2
import numpy as np
import tensorflow as tf
from keras.src.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data_dir = "Data"
categories = ["blood", "noblood"]

X, y = [], []

for category in categories:
    path = os.path.join(data_dir, category)

    if not os.path.exists(path):
        print(f"❌ Thư mục không tồn tại: {path}")
        continue

    label = categories.index(category)

    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)

        if not os.path.isfile(img_path):  # Bỏ qua file không phải ảnh
            print(f"⚠️ Bỏ qua file không hợp lệ: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Không thể đọc ảnh: {img_path}")
            continue

        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(label)


X = np.array(X, dtype='float32') / 255.0
y = np.array(y, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Kích thước ảnh
IMG_SIZE = 224
BATCH_SIZE = 32
# Load MobileNetV2 pre-trained
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Đóng băng model

# Thêm các lớp fully connected
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

y_pred = model.predict(X_test)
y_pred_classes = (y_pred>0.5).astype(int)
y_test_classes = y_test.astype(int)

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

cm = confusion_matrix(y_test_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Lưu model
model.save("blood.h5")
