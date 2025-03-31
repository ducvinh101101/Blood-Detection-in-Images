import os
import numpy as np
import cv2
import tensorflow as tf
import spectral.io.envi as envi
from keras.src.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Định nghĩa đường dẫn dữ liệu
hsi_dataset_path = "./Data_Blood_Nude/HyperBlood/data/"
blood_dataset_path = "./Data_Blood_Nude/Blood_Dataset/"


def get_good_indices():
    return np.setdiff1d(np.arange(5, 121), [43, 44, 45])


def preprocess_hsi_image(image_path):
    hdr_path = f'{image_path}.hdr'
    float_path = f'{image_path}.float'
    if not os.path.exists(hdr_path) or not os.path.exists(float_path):
        return None
    hsimage = envi.open(hdr_path, float_path)
    data = np.asarray(hsimage[:, :, :], dtype=np.float32)
    wavs = np.asarray(hsimage.bands.centers)
    good_indices = get_good_indices()
    good_indices = good_indices[good_indices < data.shape[2]]
    data = data[:, :, good_indices]
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    wavs = wavs[good_indices]

    # Trích xuất màu đỏ đặc trưng của máu
    red_band = data[:, :, np.argmin(np.abs(wavs - 600))]
    binary_mask = (red_band > 0.5).astype(np.uint8) * 255
    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    return cv2.resize(binary_mask, (224, 224))


def preprocess_rgb_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.resize(image, (224, 224)).astype(np.float32) / 255.0


def load_data():
    X, y = [], []

    # Xử lý ảnh HSI thành ảnh nhị phân để hỗ trợ nhận diện màu đỏ
    for file in os.listdir(hsi_dataset_path):
        if file.endswith(".float"):
            img = preprocess_hsi_image(os.path.join(hsi_dataset_path, file.replace(".float", "")))
            if img is not None:
                X.append(img)
                y.append(1)  # Ảnh HSI được coi là ảnh chứa máu

    # Thêm dữ liệu từ Blood_Dataset
    for category, label in [('blood', 1), ('noblood', 0)]:
        category_path = os.path.join(blood_dataset_path, category)
        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                img = preprocess_rgb_image(os.path.join(category_path, file))
                if img is not None:
                    X.append(img)
                    y.append(label)

    return np.array(X), np.array(y)


# Tải dữ liệu
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Xây dựng mô hình CNN với EfficientNetB0
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Huấn luyện mô hình
base_model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
gen = ImageDataGenerator(rotation_range=30, horizontal_flip=True, brightness_range=[0.7, 1.3],
                         shear_range=0.3, width_shift_range=0.2, height_shift_range=0.2)
train_generator = gen.flow(X_train, y_train, batch_size=16)

model.fit(train_generator, validation_data=(X_test, y_test), epochs=30)

# Fine-tune mô hình
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=(X_test, y_test), epochs=10)
