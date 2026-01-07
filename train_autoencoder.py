import numpy as np
import cv2
import os

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from models.image_autoencoder import ImageAutoencoder


# ----------------- build model -----------------
auto = ImageAutoencoder(latent_channels=8)


# ----------------- load dataset -----------------
(x_train, _), (_, _) = cifar10.load_data()

images = []

for img in x_train[:8000]:
    img = cv2.resize(img, (64, 64))
    images.append(img)

images = np.array(images, dtype="float32") / 255.0

print("Training images:", images.shape)


# ----------------- AUTO RESUME -----------------
WEIGHTS = "models/cnn_autoencoder.weights.h5"

if os.path.exists(WEIGHTS):
    try:
        auto.autoencoder.load_weights(WEIGHTS)
        print("✅ Previous checkpoint found — resuming training")
    except Exception as e:
        print("⚠️ Checkpoint exists but incompatible, starting fresh")
        print(e)
else:
    print("🆕 No checkpoint found — starting new training")


# ----------------- callbacks -----------------
checkpoint = ModelCheckpoint(
    WEIGHTS,
    save_best_only=True,
    save_weights_only=True,
    monitor="loss",
    verbose=1
)

log = CSVLogger("training_log.csv", append=True)


# ----------------- train -----------------
auto.autoencoder.fit(
    images,
    images,
    epochs=40,
    batch_size=32,
    shuffle=True,
    callbacks=[checkpoint, log]
)


# ----------------- save full model -----------------
auto.autoencoder.save("models/cnn_autoencoder.keras")

print("🎉 Training complete")
