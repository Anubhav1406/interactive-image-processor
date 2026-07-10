# PixelForge

An interactive image editor built **from scratch on OpenCV + Streamlit**. Every
operation is a small pure function in `core/`, and each function's docstring
explains the **math behind the tool** — that's the whole point of the project:
understand the maths, then implement it.

> Rename the app in one place: the `APP_NAME` constant at the top of `ui/app.py`.

## Run

The virtual environment lives in `./venv`.

```bash
# option A — activate (note: activate is *sourced*, not executed)
source venv/bin/activate
pip install -r requirements.txt        # first time only
streamlit run ui/app.py

# option B — no activation needed, call the venv's python directly
./venv/bin/python -m streamlit run ui/app.py
```

Then open http://localhost:8501 (Streamlit's default port).

**If launch hangs with an empty log**, Streamlit's file watcher is stuck on
this path. Start with the watcher disabled:

```bash
./venv/bin/python -m streamlit run ui/app.py --server.fileWatcherType none
```

## Features

**Adjustments** (live sliders)
- Auto white balance (gray-world), saturation, grayscale
- Brightness, contrast, gamma, local contrast (CLAHE)
- Skin smoothing, sharpen, blur

**Geometry & sizing**
- Rotate, flip, crop-to-aspect (passport 35:45, square 1:1, 3:4, 2:3)
- Exact-pixel resize

**Cursor tools** (pick one under *Canvas tool*)
- **Crop** — drag a rectangle on the photo
- **Brush** — paint colour, heal blemishes, or paint blur/sharpen/grayscale
  onto just the area you drag over

**Export**
- PNG / JPEG, custom filename
- Optional file-size budget (binary-search JPEG quality — handy for portals
  that cap uploads at, say, 200 KB)

## Architecture

```
core/        one file per family of operations; each apply_*(img, ...) -> BGR uint8
  grayscale.py            luminance conversion                         (original)
  brightness_contrast.py  additive brightness, pivot-128 contrast      (original)
  color.py                gamma, saturation (HSV), gray-world white balance
  histogram.py            histogram equalization + CLAHE (CDF maths)
  detail.py               Gaussian blur, unsharp-mask sharpen
  retouch.py              bilateral skin smoothing, inpainting heal
  geometry.py             rotate, flip, crop-to-aspect, crop_box, resize
  brush.py                blend_by_alpha (masked blend), strokes_to_mask
  export.py               encode to bytes, binary-search JPEG to a KB budget
ui/
  app.py                  Streamlit UI: sidebar controls -> pipeline -> preview + export
  canvas_compat.py        shim so streamlit-drawable-canvas works on modern Streamlit
```

The UI is **stateless**: on every interaction it re-derives the processed image
from a fresh copy of the original by running the pipeline in a fixed order —

```
white balance -> brightness -> contrast -> gamma -> CLAHE -> saturation
             -> skin smooth -> blur -> sharpen
             -> rotate -> flip -> crop -> resize -> grayscale -> brush
```

No accumulated state, no drift: the same slider values always produce the same
result.

## The maths, tool by tool

| Tool | Idea |
|------|------|
| Gamma | `out = (in/255)^(1/γ)·255` — non-linear curve moves midtones, fixes exposure |
| Saturation | scale the S channel in HSV, leave hue & brightness alone |
| White balance | gray-world: rescale each B/G/R channel so their means match |
| CLAHE | equalize the tone histogram per tile via its local CDF, clip-limited |
| Sharpen | unsharp mask: `img + α·(img − blur)` — add back the edges blur removed |
| Skin smooth | bilateral filter: blur weighted by distance **and** colour similarity |
| Crop-to-aspect | centre-crop to a ratio; trims, never stretches |
| Export budget | binary-search JPEG quality (~7 encodes) to fit a KB cap |
| Brushes | masked blend: `out = alpha·overlay + (1−alpha)·base` |
| Heal | `cv2.inpaint` (Telea Fast Marching) rebuilds a spot from surrounding pixels |

## Cursor tools in depth

### Interactive crop
*Canvas tool → Crop*, then drag a rectangle. The box is drawn on a downscaled
*display* copy of the image and mapped back to full-resolution pixels before the
actual slice (`core/geometry.py:crop_box`), so you always crop at full quality.

### Brushes
*Canvas tool → Brush*, choose a brush, and drag on the photo. Four of the five
brushes are the **same operation** — a masked blend (`core/brush.py:blend_by_alpha`):

    out = alpha * overlay + (1 - alpha) * base

The canvas hands back your strokes as an RGBA image; its alpha channel is the
brush mask. Only the `overlay` changes per brush (painted colour / blurred image
/ sharpened image / grayscale). The effect is computed on the whole image once,
and the mask decides where it lands.

**Heal** is the exception: inpainting *replaces* pixels rather than blending an
overlay, so it uses a hard 0/255 mask (`core/retouch.py:apply_heal`) instead of
a soft alpha blend.

## The canvas compatibility shim

`streamlit-drawable-canvas` (the library behind Crop and Brush) targets an older
Streamlit and calls `streamlit.elements.image.image_to_url` with a signature
that Streamlit ≥ 1.40 changed — so it imports fine but crashes the moment you
draw. `ui/canvas_compat.py` re-exposes the old call and adapts it to the new
API, with no Streamlit downgrade. Call `patch_image_to_url()` once before the
first `st_canvas(...)` (already wired up in `app.py`).

## Dependencies

See `requirements.txt`: `streamlit`, `opencv-python`, `numpy`, `Pillow`,
`streamlit-drawable-canvas`.
