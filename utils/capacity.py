import numpy as np

def calculate_capacity(image) -> int:
    """Calculate maximum characters that can be embedded"""
    if isinstance(image, str):
        import cv2
        image = cv2.imread(image)
    height, width, channels = image.shape
    total_bits = height * width * channels
    # 8 bits per char + 32 bits for length header
    max_chars = (total_bits - 32) // 8
    return max_chars

def check_capacity(image, message: str, encrypted: bool = False) -> tuple:
    """Check if message fits in image - returns (can_fit, msg_len, max_cap)"""
    max_chars = calculate_capacity(image)
    msg_len = len(message)
    if encrypted:
        msg_len += 32  # AES overhead
    return (msg_len <= max_chars, msg_len, max_chars)  # Fixed: now returns 3 values

def get_capacity_info(image) -> str:
    """Get human-readable capacity information"""
    max_chars = calculate_capacity(image)
    return f"Maximum capacity: {max_chars:,} characters (~{max_chars // 1000}KB)"
