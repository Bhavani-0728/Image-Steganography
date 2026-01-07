import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ------------------ BASIC METRICS ------------------

def calculate_psnr(original: np.ndarray, stego: np.ndarray) -> float:
    """
    PSNR = imperceptibility metric (higher is better).
    """
    return peak_signal_noise_ratio(original, stego, data_range=255)


def calculate_ssim(original: np.ndarray, stego: np.ndarray) -> float:
    """
    SSIM = structural similarity (1.0 is perfect).
    """
    return structural_similarity(
        original,
        stego,
        channel_axis=2,
        data_range=255
    )


# ------------------ SECRET IMAGE RECOVERY METRICS ------------------

def calculate_recovery_metrics(original_secret: np.ndarray,
                               recovered_secret: np.ndarray) -> dict:
    """
    Evaluate how well the secret image was reconstructed
    from stego image using autoencoder.
    """
    psnr = peak_signal_noise_ratio(original_secret, recovered_secret, data_range=255)

    ssim = structural_similarity(
        original_secret,
        recovered_secret,
        channel_axis=2,
        data_range=255
    )

    return {
        "psnr": psnr,
        "ssim": ssim
    }


# ------------------ HISTOGRAM COMPARISON ------------------

def plot_histogram_comparison(original: np.ndarray,
                              stego: np.ndarray,
                              title1="Original",
                              title2="Stego",
                              save_path: str = None):

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    colors = ['Red', 'Green', 'Blue']

    for i, color in enumerate(colors):
        # original
        axes[0, i].hist(original[:, :, i].flatten(), bins=256,
                        color=color.lower(), alpha=0.7)
        axes[0, i].set_title(f'{title1} - {color}')

        # stego
        axes[1, i].hist(stego[:, :, i].flatten(), bins=256,
                        color=color.lower(), alpha=0.7)
        axes[1, i].set_title(f'{title2} - {color}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


# ------------------ DIFFERENCE MAP ------------------

def difference_map(original: np.ndarray, stego: np.ndarray) -> np.ndarray:
    """
    Returns visually amplified difference image for plotting/report.
    """
    diff = np.abs(original.astype("int16") - stego.astype("int16"))
    diff = np.clip(diff * 20, 0, 255).astype("uint8")
    return diff
