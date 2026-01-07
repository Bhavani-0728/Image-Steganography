import tensorflow as tf
from tensorflow.keras import layers, models


class ImageAutoencoder:

    def __init__(self, latent_channels=8):
        self.latent_channels = latent_channels
        self.encoder, self.decoder, self.autoencoder = self.build()

    def build(self):

        inp = layers.Input(shape=(64, 64, 3))   # fixed training size

        # ------------ ENCODER ------------
        c1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
        p1 = layers.MaxPool2D()(c1)

        c2 = layers.Conv2D(64, 3, padding="same", activation="relu")(p1)
        p2 = layers.MaxPool2D()(c2)

        bottleneck = layers.Conv2D(
            self.latent_channels,
            3,
            padding="same",
            activation="relu"
        )(p2)

        # ------------ DECODER (with skip connections) ------------
        u1 = layers.UpSampling2D()(bottleneck)
        m1 = layers.Concatenate()([u1, c2])
        c3 = layers.Conv2D(64, 3, padding="same", activation="relu")(m1)

        u2 = layers.UpSampling2D()(c3)
        m2 = layers.Concatenate()([u2, c1])
        c4 = layers.Conv2D(32, 3, padding="same", activation="relu")(m2)

        out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(c4)

        auto = models.Model(inp, out)

        auto.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mse"]
        )

        encoder = models.Model(inp, bottleneck)
        decoder = None  # not needed separately here

        return encoder, None, auto

    # ---------------- INFERENCE HELPERS ----------------

    def encode(self, img):
        img = cv2.resize(img, (64, 64))
        img = img.astype("float32") / 255.0
        return self.encoder.predict(img[None, ...], verbose=0)[0]

    def decode_from_auto(self, latent):
        pass  # optional if you want separate decode
