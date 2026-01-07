import numpy as np
import cv2
import os

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint

from models.image_autoencoder import ImageAutoencoder


auto = ImageAutoencoder(latent_channels=8)   


# ---------- LOAD CIFAR10 ----------
(x_train, _), (_, _) = cifar10.load_data()

images = []

for img in x_train[:8000]:
    img = cv2.resize(img, (64, 64))      
    images.append(img)

images = np.array(images, dtype="float32") / 255.0

print("Training images:", images.shape)


# ---------- CHECKPOINT ----------
checkpoint = ModelCheckpoint(
    "models/cnn_autoencoder.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="loss",
    verbose=1
)


# ---------- TRAIN ----------
auto.autoencoder.fit(
    images,
    images,
    epochs=25,
    batch_size=32,
    shuffle=True,
    callbacks=[checkpoint]
)


auto.autoencoder.save("models/cnn_autoencoder.keras")

print("Training finished successfully.")
