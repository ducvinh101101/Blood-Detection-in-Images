from keras import Sequential
from keras.api.applications import VGG16
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.applications.resnet import ResNet50
from keras.src.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.callbacks import ModelCheckpoint
from keras.api.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# Định nghĩa đường dẫn dữ liệu
data_dir = "D:/PyCharmProjects/Blood/Data_Blood_Nude/datanew"

# Tạo ImageDataGenerator với tiền xử lý mở rộng dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
train_generator = train_datagen.flow_from_directory(
    data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True, subset='training'
)
val_generator = train_datagen.flow_from_directory(
    data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation'
)

print(f"Train size: {train_generator.samples}, Validation size: {val_generator.samples}")

# Load mô hình VGG16 pre-trained
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  # Đóng băng các layer của VGG16

# Xây dựng model
top_model = Sequential([
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Kết hợp model
model = Sequential([base_model, top_model])

# Biên dịch mô hình
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Callback để lưu mô hình tốt nhất
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)

# Huấn luyện mô hình
H = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=1,
    callbacks=[checkpoint]
)

# Vẽ biểu đồ độ chính xác
plt.figure(figsize=(10, 5))
plt.plot(H.history['accuracy'], label='Train Accuracy')
plt.plot(H.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Dự đoán và vẽ Confusion Matrix
y_true = []
y_pred = []
for i in range(len(val_generator)):
    x_batch, y_batch = val_generator[i]
    preds = model.predict(x_batch)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
    if len(y_true) >= val_generator.samples:
        break

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# In báo cáo phân loại
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))