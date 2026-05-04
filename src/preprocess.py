"""
Image loading and preprocessing for the Hunyuan3D multiview pipeline.

Covers plan Steps A and B:
  A — Load and validate one or more input images (RGBA).
  B — Background removal (rembg) + neutral composite for paint conditioning.

Design decisions (from the consolidated plan):
  - Gray composite (127) for the paint model reference:  reduces directional
    lighting bias, giving more balanced texture generation.
  - White composite for the shape-model input:  DiT was trained on white-bg images.
  - Alpha-extrema check before running rembg:  don't destroy pre-existing masks.
  - Lazy BackgroundRemover import:  avoids onnxruntime overhead when not needed.
  - EXIF rotation not applied by default (correct for synthetic renders;
    pass exif_transpose=True for real photographs).

All heavy imports (rembg, numpy) are deferred to function bodies so that the
module is importable on macOS without GPU dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step A: loading and validation
# ---------------------------------------------------------------------------

def load_image_rgba(
    path: Path | str,
    *,
    exif_transpose: bool = False,
) -> Image.Image:
    """
    Open an image file and return it as RGBA.

    Args:
        path: Path to the image file.
        exif_transpose: Apply EXIF orientation correction.
                        Useful for photographs; leave False for renders.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image is fully transparent (no subject found).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)

    if exif_transpose:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)

    img = img.convert("RGBA")

    if img.getbbox() is None:
        raise ValueError(f"Image is fully transparent (empty bounding box): {path}")

    logger.debug("Loaded %s — size=%s mode=RGBA", path.name, img.size)
    return img


def collect_views(
    image: Optional[str | Path] = None,
    front: Optional[str | Path] = None,
    left: Optional[str | Path] = None,
    right: Optional[str | Path] = None,
    back: Optional[str | Path] = None,
    *,
    exif_transpose: bool = False,
) -> dict[str, Image.Image]:
    """
    Build the view dictionary for geometry generation.

    Single-image mode  (Path A): pass ``image`` or ``front``.
    Four-view mode     (Path B): pass ``front``, ``left``, ``right``, ``back``.

    Returns:
        dict with keys from {"front", "left", "right", "back"}, values are RGBA images.

    Raises:
        ValueError: If no images are provided.
    """
    views: dict[str, Image.Image] = {}

    for key, val in (("front", front or image), ("left", left), ("right", right), ("back", back)):
        if val is not None:
            views[key] = load_image_rgba(val, exif_transpose=exif_transpose)

    if not views:
        raise ValueError(
            "Provide at least one image via --image / --front (single-image mode) "
            "or --front + --left + --right + --back (four-view mode)."
        )

    return views


# ---------------------------------------------------------------------------
# Step B: background removal + compositing
# ---------------------------------------------------------------------------

def maybe_remove_background(
    image: Image.Image,
    remove_bg: bool,
    background_remover=None,
) -> Image.Image:
    """
    Run background removal only when necessary.

    Background removal is **skipped** if the image already has a non-trivial
    alpha channel (alpha_min < 255).  This prevents destroying pre-existing masks
    from rendering pipelines or manual editing.

    Args:
        image:               Input image (any mode; converted to RGBA internally).
        remove_bg:           Master switch — set False to skip entirely.
        background_remover:  Optional pre-loaded BackgroundRemover instance.
                             If None and removal is needed, one is created lazily.

    Returns:
        RGBA image with background removed (or original RGBA if skipped).
    """
    image = image.convert("RGBA")

    if not remove_bg:
        return image

    alpha = image.getchannel("A")
    alpha_min, alpha_max = alpha.getextrema()

    if alpha_min == alpha_max == 255:
        # Fully opaque — safe to run removal.
        if background_remover is None:
            from hy3dshape.rembg import BackgroundRemover  # type: ignore[import]
            background_remover = BackgroundRemover()
        logger.debug("Running background removal (fully opaque input).")
        image = background_remover(image)
    else:
        logger.debug(
            "Skipping background removal — image already has transparency "
            "(alpha_min=%d, alpha_max=%d).",
            alpha_min,
            alpha_max,
        )

    return image


def compose_over_gray(image: Image.Image, gray: int = 127) -> Image.Image:
    """
    Composite an RGBA image onto a neutral gray background and return RGB.

    Gray (≈ 127) is used as the paint-model appearance reference because it
    minimises directional lighting bias compared with a white background,
    giving the multiview diffusion model a more balanced starting point.

    Args:
        image: Input image (any mode; converted to RGBA internally).
        gray:  Background luminance, 0–255.  Default 127 matches the plan.

    Returns:
        RGB image composited over the gray background.
    """
    image = image.convert("RGBA")
    bg = Image.new("RGBA", image.size, (gray, gray, gray, 255))
    composed = Image.alpha_composite(bg, image)
    return composed.convert("RGB")


def compose_over_white(image: Image.Image) -> Image.Image:
    """
    Composite an RGBA image onto white and return RGB.

    Used for **shape-model conditioning** — the DiT was trained with white-bg images.

    Args:
        image: Input image (any mode; converted to RGBA internally).

    Returns:
        RGB image composited over a white background.
    """
    image = image.convert("RGBA")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(bg, image)
    return composed.convert("RGB")


# ---------------------------------------------------------------------------
# Convenience: full preprocess pipeline for a single view
# ---------------------------------------------------------------------------

def preprocess_view(
    image: Image.Image,
    *,
    remove_bg: bool = True,
    background_remover=None,
    for_shape: bool = False,
    gray: int = 127,
) -> Image.Image:
    """
    Run the full preprocessing chain for one view image.

    Steps:
        1. Maybe remove background (rembg).
        2. Composite over gray (paint reference) or white (shape input).

    Args:
        image:               Input image.
        remove_bg:           Whether to run background removal.
        background_remover:  Pre-loaded remover instance (reuse across views).
        for_shape:           If True, composite over white (shape conditioning).
                             If False, composite over gray (paint reference).
        gray:                Gray level for paint composite.

    Returns:
        RGB image ready for model input.
    """
    rgba = maybe_remove_background(image, remove_bg=remove_bg, background_remover=background_remover)
    if for_shape:
        return compose_over_white(rgba)
    return compose_over_gray(rgba, gray=gray)


def preprocess_all_views(
    views: dict[str, Image.Image],
    *,
    remove_bg: bool = True,
) -> dict[str, dict[str, Image.Image]]:
    """
    Preprocess all collected view images and return both variants.

    Lazily creates a single BackgroundRemover and reuses it across views
    to avoid loading the ONNX model repeatedly.

    Args:
        views:      Dict of view name → RGBA PIL image (from collect_views).
        remove_bg:  Whether to run background removal.

    Returns:
        Dict mapping each view name to a sub-dict:
            {
                "shape":  RGB composited over white  (for shape DiT),
                "paint":  RGB composited over gray   (for paint diffusion),
                "rgba":   RGBA after background removal (for intermediates),
            }
    """
    background_remover = None

    if remove_bg:
        try:
            from hy3dshape.rembg import BackgroundRemover  # type: ignore[import]
            background_remover = BackgroundRemover()
            logger.info("BackgroundRemover loaded.")
        except ImportError:
            logger.warning(
                "hy3dshape.rembg not importable — skipping background removal. "
                "This is expected on macOS without the full Hunyuan3D-2.1 install."
            )
            remove_bg = False

    result: dict[str, dict[str, Image.Image]] = {}

    for name, img in views.items():
        rgba = maybe_remove_background(img, remove_bg=remove_bg, background_remover=background_remover)
        result[name] = {
            "rgba": rgba,
            "shape": compose_over_white(rgba),
            "paint": compose_over_gray(rgba),
        }
        logger.debug("Preprocessed view '%s'.", name)

    return result
