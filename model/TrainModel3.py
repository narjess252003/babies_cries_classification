#CNN model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
data_directory = "C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/spectrograms/"
classes = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
img_size = 128
#loading dataset
def load_data():
    X, y = [], []
    for idx, label in enumerate(classes):
        class_path = os.path.join(data_directory, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #img in grayscale
            img = cv2.resize(img, (img_size, img_size))  #resize image 128x128 
            X.append(img)
            y.append(idx)
    return np.array(X) / 255.0, np.array(y)  
X, y = load_data()
X = X.reshape(-1, img_size, img_size, 1)  #reshape for CNN
#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#augment data 
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)
#improved CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),  
    Dense(len(classes), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#train with data augmentation
print("Train CNN")
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))
#evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"CNN accuracy: {test_acc:.4f}")
#save the model
model.save("C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/model/cnn_model.h5")
print("CNN saved")
