import numpy as np


def embed_lsb(image: np.ndarray, binary_data: str) -> np.ndarray:
    stego = image.copy().astype(np.int16)
    flat = stego.flatten()

    # prepend length header (32 bits)
    data_len = len(binary_data)
    header = format(data_len, '032b')
    full = header + binary_data

    if len(full) > len(flat):
        raise ValueError("Cover image too small to hold secret data.")

    for i, bit in enumerate(full):
        flat[i] = (flat[i] & ~1) | int(bit)

    flat = np.clip(flat, 0, 255).astype(np.uint8)
    return flat.reshape(image.shape)


def extract_lsb(image: np.ndarray) -> str:
    flat = image.flatten()

    # read length
    header = ''.join(str(flat[i] & 1) for i in range(32))
    data_len = int(header, 2)

    # read payload
    data = ''.join(str(flat[i] & 1) for i in range(32, 32 + data_len))
    return data
