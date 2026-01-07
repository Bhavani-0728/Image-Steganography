import tensorflow as tf
from tensorflow.keras import layers, models


class ImageAutoencoder:

    def __init__(self, latent_channels=8):     
        self.latent_channels = latent_channels
        self.encoder, self.decoder, self.autoencoder = self.build()

    def build(self):

        # ---------- ENCODER ----------
        inp = layers.Input(shape=(None, None, 3))

        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
        x = layers.MaxPool2D()(x)

        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPool2D()(x)

        latent = layers.Conv2D(
            self.latent_channels,
            3,
            padding="same",
            activation="relu",
            name="latent_space"
        )(x)

        encoder = models.Model(inp, latent, name="encoder")

        # ---------- DECODER ----------
        dec_in = layers.Input(shape=(None, None, self.latent_channels))

        x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(dec_in)
        x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

        out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)

        decoder = models.Model(dec_in, out, name="decoder")

        # ---------- AUTOENCODER ----------
        out_full = decoder(encoder(inp))
        auto = models.Model(inp, out_full)

        auto.compile(optimizer="adam", loss="mse")

        return encoder, decoder, auto

    def encode(self, img):
        img = img.astype("float32") / 255.0
        latent = self.encoder.predict(img[None, ...], verbose=0)[0]
        return latent.astype("float32")

    def decode(self, latent):
        result = self.decoder.predict(latent[None, ...], verbose=0)[0]
        return (result * 255).clip(0, 255).astype("uint8")
