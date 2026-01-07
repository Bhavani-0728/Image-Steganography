import numpy as np
import cv2

from core.lsb import embed_lsb
from models.image_autoencoder import ImageAutoencoder
from core.encryption import AESCipher


class ImageStegoEncoder:

    def __init__(self):
        self.model = ImageAutoencoder()
        self.model.autoencoder.load_weights("models/cnn_autoencoder.weights.h5")

    def encode(self, cover_img, secret_img, password: str = None):

        # ensure RGB
        secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2RGB)

        # ---- autoencoder latent ----
        latent = self.model.encode(secret_img)

        h, w, c = latent.shape

        # ---- save latent shape ----
        shape_bits = (
            format(h, '016b')
            + format(w, '016b')
            + format(c, '016b')
        )

        # ---- latent → bytes ----
        latent_bytes = latent.astype(np.float32).tobytes()

        # ---- encrypt if password provided ----
        if password and len(password) > 0:
            cipher = AESCipher(password)
            latent_bytes = cipher.encrypt_bytes(latent_bytes)

        # ---- bytes → bits ----
        payload_bits = ''.join(format(b, '08b') for b in latent_bytes)

        # ---- header + payload ----
        binary_data = shape_bits + payload_bits

        # ---- capacity check ----
        if len(binary_data) > cover_img.size:
            raise ValueError("Cover image does not have sufficient capacity.")

        # ---- LSB embed ----
        stego = embed_lsb(cover_img, binary_data)

        return stego
