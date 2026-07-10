"""Compatibility shim for streamlit-drawable-canvas on modern Streamlit.

Why this exists
---------------
`streamlit-drawable-canvas` (0.9.x, last released 2023) renders a background
image by calling:

    streamlit.elements.image.image_to_url(image, width, clamp, channels,
                                           output_format, image_id)

Streamlit >= ~1.40 MOVED that function to
`streamlit.elements.lib.image_utils.image_to_url` AND changed its 2nd argument
from a plain `width: int` to a `layout_config: LayoutConfig` object. So on
Streamlit 1.59 the canvas imports fine but blows up the instant you draw.

The fix
-------
Re-expose an `image_to_url` on the OLD module path that adapts the OLD call
(with an int width) to the NEW function. The new function only ever reads
`layout_config.width`, so a minimal stand-in object is enough — we don't need
to import Streamlit's internal LayoutConfig class (which keeps this shim from
breaking again on the next refactor).

Call `patch_image_to_url()` once, before the first `st_canvas(...)`.
"""
import types
import streamlit.elements.image as _st_image


def patch_image_to_url():
    # Already patched, or an old Streamlit that still has it -> nothing to do.
    if hasattr(_st_image, "image_to_url"):
        return

    from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url

    def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
        # The canvas passes an int `width`; the new API wants an object whose
        # `.width` attribute drives an optional down-resize. The background is
        # already sized to `width` by the caller, so this is effectively a no-op
        # resize and just satisfies the new signature.
        layout_config = types.SimpleNamespace(width=width)
        return _new_image_to_url(
            image, layout_config, clamp, channels, output_format, image_id
        )

    _st_image.image_to_url = _image_to_url_compat
