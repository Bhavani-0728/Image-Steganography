from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes


class AESCipher:
    def __init__(self, password: str):
        # derive strong 256-bit key
        self.key = PBKDF2(password.encode(), b"steganography_salt", dkLen=32)

    def encrypt_bytes(self, data: bytes) -> bytes:
        iv = get_random_bytes(16)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        padded = self._pad(data)
        encrypted = cipher.encrypt(padded)
        return iv + encrypted

    def decrypt_bytes(self, data: bytes) -> bytes:
        iv, encrypted = data[:16], data[16:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(encrypted)
        return self._unpad(decrypted)

    def _pad(self, data: bytes) -> bytes:
        pad_len = 16 - (len(data) % 16)
        return data + bytes([pad_len]) * pad_len

    def _unpad(self, data: bytes) -> bytes:
        return data[:-data[-1]]
