import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
import streamlit as st
from PIL import Image

# image operations (the maths for each is documented in its module docstring)
from core.grayscale import apply_grayscale
from core.brightness_contrast import apply_brightness, apply_contrast
from core.color import apply_gamma, apply_saturation, apply_white_balance
from core.histogram import apply_clahe
from core.detail import apply_gaussian_blur, apply_sharpen
from core.retouch import apply_skin_smooth, apply_heal
from core.geometry import apply_rotate, apply_flip, crop_to_aspect, crop_box, resize_to
from core.brush import blend_by_alpha, strokes_to_mask
from core.export import encode_image

# Interactive drag-to-crop needs streamlit-drawable-canvas, which breaks on
# modern Streamlit without a shim (see ui/canvas_compat.py). Guard the import so
# the rest of the app still runs if the package is missing.
CANVAS_OK = True
try:
    from ui.canvas_compat import patch_image_to_url
    patch_image_to_url()
    from streamlit_drawable_canvas import st_canvas
except Exception:
    CANVAS_OK = False

APP_NAME = "PixelForge"
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(f"🛠️ {APP_NAME}")
st.caption("An image editor built from scratch on OpenCV.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Aspect-ratio presets useful for placement / ID photos.
ASPECT_PRESETS = {
    "Free (no crop)": None,
    "Passport 35:45 (India)": (35, 45),
    "Square 1:1 (LinkedIn / portal)": (1, 1),
    "Portrait 3:4": (3, 4),
    "Photo 2:3": (2, 3),
}

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)   # BGR uint8

    # ----------------------------------------------------------------- controls
    with st.sidebar:
        st.header("Adjustments")

        # One canvas at a time: pick which cursor tool is live.
        canvas_tool = st.radio(
            "🖱️ Canvas tool",
            ["None", "Crop", "Brush"],
            horizontal=True,
            disabled=not CANVAS_OK,
            help="Crop = drag a box. Brush = paint colour or effects onto the photo.",
        )
        if not CANVAS_OK:
            st.caption("Install `streamlit-drawable-canvas` to enable Crop/Brush.")

        BRUSH_TYPES = ["Paint (colour)", "Heal (remove blemish)", "Blur", "Sharpen", "Grayscale"]
        with st.expander("🖌️ Brush tools", expanded=(canvas_tool == "Brush")):
            brush_type = st.selectbox("Brush", BRUSH_TYPES)
            brush_size = st.slider("Brush size (px)", 1, 80, 20)
            brush_color = st.color_picker("Paint colour", "#FF0000")
            brush_strength = st.slider("Effect strength", 1, 30, 12,
                                       help="Blur radius / sharpen amount for effect brushes")

        with st.expander("Colour & White Balance", expanded=True):
            auto_wb = st.checkbox("Auto white balance (gray-world)")
            saturation = st.slider("Saturation", 0.0, 3.0, 1.0, 0.05)
            grayscale = st.checkbox("Grayscale")

        with st.expander("Exposure & Contrast", expanded=True):
            brightness = st.slider("Brightness", -100, 100, 0)
            contrast = st.slider("Contrast", 0.5, 3.0, 1.0, 0.05)
            gamma = st.slider("Gamma", 0.1, 3.0, 1.0, 0.05)
            clahe = st.slider("Local contrast (CLAHE)", 0.0, 5.0, 0.0, 0.1)

        with st.expander("Detail & Retouch"):
            skin = st.slider("Skin smoothing", 0, 100, 0)
            sharpen = st.slider("Sharpen", 0.0, 3.0, 0.0, 0.1)
            blur = st.slider("Blur", 0, 25, 0)

        with st.expander("Geometry"):
            rotate = st.slider("Rotate (°)", -45.0, 45.0, 0.0, 0.5)
            flip_h = st.checkbox("Flip horizontal")
            flip_v = st.checkbox("Flip vertical")
            aspect_label = st.selectbox("Crop to aspect", list(ASPECT_PRESETS),
                                        disabled=(canvas_tool == "Crop"),
                                        help="Ignored while the Crop canvas tool is active.")

        with st.expander("Resize"):
            do_resize = st.checkbox("Resize to exact pixels")
            out_w = st.number_input("Width (px)", 1, 10000, image.shape[1])
            out_h = st.number_input("Height (px)", 1, 10000, image.shape[0])

    # --------------------------------------------------------------- pipeline
    # Order matters: colour/exposure first, then retouch/detail, then geometry.
    processed = image.copy()

    if auto_wb:
        processed = apply_white_balance(processed)

    processed = apply_brightness(processed, brightness)
    processed = apply_contrast(processed, contrast)
    processed = apply_gamma(processed, gamma)
    processed = apply_clahe(processed, clahe)
    processed = apply_saturation(processed, saturation)

    processed = apply_skin_smooth(processed, skin)
    processed = apply_gaussian_blur(processed, blur)
    processed = apply_sharpen(processed, sharpen)

    processed = apply_rotate(processed, rotate)
    processed = apply_flip(processed, flip_h, flip_v)

    # ---- cropping: interactive drag-box OR aspect preset (mutually exclusive)
    if canvas_tool == "Crop" and CANVAS_OK:
        st.subheader("✏️ Draw a crop box")
        st.caption("Drag on the image to draw a rectangle. Use the last one you draw.")

        # Downscale for display so the canvas fits the page; remember the factor
        # so we can map the drawn box back to full-resolution pixels.
        disp_w = min(processed.shape[1], 640)
        scale = disp_w / processed.shape[1]          # display px per real px
        disp_h = max(1, int(round(processed.shape[0] * scale)))
        bg = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)).resize((disp_w, disp_h))

        canvas = st_canvas(
            fill_color="rgba(0, 229, 255, 0.15)",    # translucent selection
            stroke_width=2,
            stroke_color="#00E5FF",
            background_image=bg,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            display_toolbar=True,
            key="crop_canvas",
        )

        rect = None
        if canvas.json_data and canvas.json_data.get("objects"):
            rect = canvas.json_data["objects"][-1]   # newest rectangle wins

        if rect:
            inv = 1.0 / scale                        # real px per display px
            x = rect["left"] * inv
            y = rect["top"] * inv
            w = rect["width"] * rect.get("scaleX", 1) * inv
            h = rect["height"] * rect.get("scaleY", 1) * inv
            processed = crop_box(processed, x, y, w, h)
            st.caption(f"Cropped to {processed.shape[1]}×{processed.shape[0]} px")
    else:
        ratio = ASPECT_PRESETS[aspect_label]
        if ratio is not None:
            processed = crop_to_aspect(processed, ratio[0], ratio[1])

    if do_resize:
        processed = resize_to(processed, int(out_w), int(out_h))

    if grayscale:                      # last: removing colour is destructive
        processed = apply_grayscale(processed)

    # ---- brush: paint colour or effects onto the (final) image with the cursor
    if canvas_tool == "Brush" and CANVAS_OK:
        st.subheader("🖌️ Paint on the image")
        st.caption(f"Brush: **{brush_type}** — drag on the photo below.")

        # Display-resolution canvas; map the painted mask back to full res.
        disp_w = min(processed.shape[1], 640)
        scale = disp_w / processed.shape[1]
        disp_h = max(1, int(round(processed.shape[0] * scale)))
        bg = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)).resize((disp_w, disp_h))

        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=brush_size,
            stroke_color=brush_color,
            background_image=bg,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="freedraw",
            display_toolbar=True,
            key="brush_canvas",
        )

        if canvas.image_data is not None:
            strokes = canvas.image_data                     # (disp_h, disp_w, 4) RGBA
            if strokes[:, :, 3].any():                      # something was painted
                full = (processed.shape[1], processed.shape[0])
                mask = strokes_to_mask(strokes, full)       # (H, W) alpha in [0,1]

                if brush_type == "Heal (remove blemish)":
                    # Inpainting REPLACES pixels, so it takes a hard 0/255 mask
                    # instead of a soft blend. Dilate slightly so the whole
                    # blemish (plus a rim) is covered and healed cleanly.
                    mask255 = (mask > 0.3).astype(np.uint8) * 255
                    mask255 = cv2.dilate(mask255, np.ones((3, 3), np.uint8))
                    radius = int(np.clip(brush_strength / 2, 3, 15))
                    processed = apply_heal(processed, mask255, radius)
                else:
                    if brush_type == "Paint (colour)":
                        # Painted RGB itself is the overlay (convert RGBA->BGR).
                        rgb = strokes[:, :, :3][:, :, ::-1]
                        overlay = cv2.resize(rgb, full, interpolation=cv2.INTER_LINEAR)
                    elif brush_type == "Blur":
                        overlay = apply_gaussian_blur(processed, brush_strength)
                    elif brush_type == "Sharpen":
                        overlay = apply_sharpen(processed, brush_strength / 5.0)
                    else:  # Grayscale
                        overlay = apply_grayscale(processed)

                    processed = blend_by_alpha(processed, overlay, mask)

    # ----------------------------------------------------------------- preview
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=f"Original — {image.shape[1]}×{image.shape[0]}", channels="BGR")
    with col2:
        st.image(processed, caption=f"Processed — {processed.shape[1]}×{processed.shape[0]}", channels="BGR")

    # ------------------------------------------------------------------ export
    st.subheader("Export")
    e1, e2, e3 = st.columns(3)
    with e1:
        file_name = st.text_input("File name", "processed_image").strip() or "processed_image"
    with e2:
        file_format = st.selectbox("Format", ["png", "jpg", "jpeg"])
    with e3:
        limit_size = st.checkbox("Limit file size (JPEG)")
        target_kb = st.number_input("Max KB", 20, 5000, 200) if limit_size else None

    data, actual_kb = encode_image(processed, file_format, target_kb)
    st.caption(f"Encoded size: {actual_kb:.1f} KB")
    st.download_button(
        label="⬇️ Download Image",
        data=data,
        file_name=f"{file_name}.{file_format}",
        mime=f"image/{file_format}",
    )
else:
    st.info("Upload a photo to begin. All processing happens locally in OpenCV.")
