# Lung Cancer Detection using CNN
# This project uses deep learning to classify lung cancer images

# -------------------------------
# Importing required libraries
# -------------------------------

import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from zipfile import ZipFile

from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')


# -------------------------------
# Loading and extracting dataset
# -------------------------------

data_path = 'lung_subset_small_folder.zip'

with ZipFile(data_path, 'r') as zip:
    zip.extractall()
    print("Dataset extracted successfully")

# folder where dataset is stored
path = 'lung_subset_small'


# -------------------------------
# Visualizing some sample images
# -------------------------------

classes = ['lung_n', 'lung_aca', 'lung_scc']

for cat in classes:
    image_dir = f'{path}/{cat}'
    images = os.listdir(image_dir)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Sample images from {cat}", fontsize=16)

    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(f'{image_dir}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')

    plt.show()


# -------------------------------
# Preparing the dataset
# -------------------------------

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10

# using ImageDataGenerator so we don't load everything into memory at once
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# training data
train_data = datagen.flow_from_directory(
    path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# validation data
val_data = datagen.flow_from_directory(
    path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# -------------------------------
# Building the CNN model
# -------------------------------

model = keras.models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')  # 3 classes
])

model.summary()


# -------------------------------
# Callbacks (to improve training)
# -------------------------------

# custom callback to stop training if accuracy crosses 90%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.90:
            print("Stopping early - reached 90% accuracy")
            self.model.stop_training = True

# stops training if model stops improving
es = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# reduces learning rate if validation loss stops improving
lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.5
)


# -------------------------------
# Compiling the model
# -------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# -------------------------------
# Training the model
# -------------------------------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[es, lr, myCallback()]
)


# -------------------------------
# Plotting training results
# -------------------------------

history_df = pd.DataFrame(history.history)

history_df[['accuracy', 'val_accuracy']].plot()
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# -------------------------------
# Evaluating the model
# -------------------------------

Y_pred = model.predict(val_data)
Y_pred_labels = np.argmax(Y_pred, axis=1)
Y_true = val_data.classes

print(metrics.classification_report(
    Y_true,
    Y_pred_labels,
    target_names=classes
))
