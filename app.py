import streamlit as st
import numpy as np
import cv2

from core.image_encoder import ImageStegoEncoder
from core.image_decoder import ImageStegoDecoder

from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
)

from utils.image_utils import (
    resize_for_autoencoder,
    resize_keep_aspect,
)

st.set_page_config(page_title="CNN Autoencoder Image Steganography")

st.title("🖼️ Autoencoder-Based Image Steganography (CNN + LSB)")

encoder = ImageStegoEncoder()
decoder = ImageStegoDecoder()

tab1, tab2 = st.tabs(["🔐 Encode Secret Image", "🔎 Decode Secret Image"])


# ================= ENCODE =================
with tab1:

    st.header("Hide Secret Image")

    cover_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"])
    secret_file = st.file_uploader("Upload Secret Image", type=["png", "jpg", "jpeg"])

    password = st.text_input("Encryption password (optional)", type="password")

    if cover_file and secret_file:

        cover_img = cv2.imdecode(np.frombuffer(cover_file.read(), np.uint8), 1)
        secret_img = cv2.imdecode(np.frombuffer(secret_file.read(), np.uint8), 1)

        # resize
        cover_img = resize_keep_aspect(cover_img, 512)
        secret_img = resize_for_autoencoder(secret_img, 64)      # <<=== 64

        st.image(
            [
                cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(secret_img, cv2.COLOR_BGR2RGB)
            ],
            caption=["Cover Image", "Secret Image (64×64 for model)"],
            use_container_width=True
        )

        if st.button("✨ Encode Secret Image"):

            try:
                stego = encoder.encode(cover_img, secret_img, password=password)

                psnr = calculate_psnr(cover_img, stego)
                ssim = calculate_ssim(cover_img, stego)

                st.success("Encoding completed!")

                st.metric("PSNR", f"{psnr:.2f} dB")
                st.metric("SSIM", f"{ssim:.4f}")

                st.image(cv2.cvtColor(stego, cv2.COLOR_BGR2RGB),
                         caption="Stego Image",
                         use_container_width=True)

                _, buff = cv2.imencode(".png", stego)

                st.download_button(
                    "⬇️ Download Stego Image",
                    data=buff.tobytes(),
                    mime="image/png",
                    file_name="stego_image.png"
                )

            except Exception as e:
                st.error(f"Encoding failed: {e}")


# ================= DECODE =================
with tab2:

    st.header("Recover Secret Image")

    stego_file = st.file_uploader("Upload Stego Image", type=["png", "jpg", "jpeg"])

    password2 = st.text_input("Password (if used during encoding)", type="password")

    if stego_file:

        stego_img = cv2.imdecode(np.frombuffer(stego_file.read(), np.uint8), 1)

        st.image(cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB),
                 caption="Stego Image",
                 use_container_width=True)

        if st.button("📥 Decode"):

            try:
                recovered = decoder.decode(stego_img, password=password2)

                st.success("Secret image recovered successfully!")

                st.image(
                    cv2.cvtColor(recovered, cv2.COLOR_BGR2RGB),
                    caption="Recovered Secret Image",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Decoding failed: {e}")
