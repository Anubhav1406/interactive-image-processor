import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.grayscale import apply_grayscale
from core.brightness_contrast import apply_brightness, apply_contrast
import streamlit as st
import numpy as np
import cv2

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if "grayscale" not in st.session_state:
    st.session_state.grayscale = False

if uploaded_file is not None:
    

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    processed = image.copy()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original", channels='BGR')

    with col3:
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", 0.5, 3.0, 1.0)

        processed = apply_contrast(processed, contrast)
        processed = apply_brightness(processed, brightness)

        if st.button("Toggle Grayscale"):
            st.session_state.grayscale = not st.session_state.grayscale

        if st.session_state.grayscale:
            processed = apply_grayscale(processed)

        file_name = st.text_input("File name", "processed_image")
        if file_name.strip() == "":
            file_name = "processed_image"
        file_format = st.selectbox("Format", ["png", "jpg", "jpeg"])

    with col2:
        st.image(processed, caption="Processed", channels='BGR')

    ext = f".{file_format}"
    _, buffer = cv2.imencode(ext, processed)
    image_bytes = buffer.tobytes()

    st.download_button(
        label="Download Image",
        data=image_bytes,
        file_name=f"{file_name}.{file_format}",
        mime=f"image/{file_format}"
    )