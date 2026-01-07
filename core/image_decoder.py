import numpy as np
import cv2

from core.lsb import extract_lsb
from models.image_autoencoder import ImageAutoencoder
from core.encryption import AESCipher


class ImageStegoDecoder:

    def __init__(self):
        self.model = ImageAutoencoder()
        self.model.autoencoder.load_weights("models/cnn_autoencoder.weights.h5")

    def decode(self, stego_img, password: str = None):

        bits = extract_lsb(stego_img)

        # ---- latent shape ----
        h = int(bits[0:16], 2)
        w = int(bits[16:32], 2)
        c = int(bits[32:48], 2)

        payload_bits = bits[48:]

        # ---- bits → encrypted bytes ----
        encrypted_bytes = bytes(
            int(payload_bits[i:i+8], 2)
            for i in range(0, len(payload_bits), 8)
        )

        # ---- decrypt if password was used ----
        if password and len(password) > 0:
            cipher = AESCipher(password)
            latent_bytes = cipher.decrypt_bytes(encrypted_bytes)
        else:
            latent_bytes = encrypted_bytes

        # ---- bytes → latent tensor ----
        latent = np.frombuffer(latent_bytes, dtype=np.float32).reshape(h, w, c)

        # ---- decode ----
        recovered = self.model.decode(latent)

        # convert back to BGR for OpenCV display if needed
        recovered = cv2.cvtColor(recovered, cv2.COLOR_RGB2BGR)

        return recovered
