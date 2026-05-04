"""
Image loading and preprocessing for the Hunyuan3D multiview pipeline.

Covers plan Steps A and B:
  A — Load and validate one or more input images (RGBA).
  B — Background removal (rembg) + neutral composite for paint conditioning.

Design decisions (from the consolidated plan):
  - Gray composite (127) for the paint model reference:  reduces directional
    lighting bias, giving more balanced texture generation.
  - White composite for the shape-model input:  DiT was trained on white-bg images.
  - rembg always runs when remove_bg=True:  even images with existing transparency
    are re-processed to ensure a clean, model-quality alpha mask.  Soft or
    garbage alpha from JPEG compression / photo-editing apps bleeds background
    colour into the gray/white composite, which distorts shape generation and
    bakes lighting artefacts into the final texture.
  - Optional alpha-edge erosion (erode_alpha):  trims halo bleed left by rembg
    around high-contrast foreground edges.
  - Lazy BackgroundRemover import:  avoids onnxruntime overhead when not needed.
  - EXIF rotation not applied by default (correct for synthetic renders;
    pass exif_transpose=True for real photographs).

All heavy imports (rembg, numpy, cv2) are deferred to function bodies so that
the module is importable on macOS without GPU dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Canonical view order shared with mesh_generate.py.
# The 2mv pipeline expects images in exactly this sequence.
VIEW_ORDER: list[str] = ["front", "left", "right", "back"]

# Image file extensions recognised when scanning a multiview folder.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
)


# ---------------------------------------------------------------------------
# Multiview folder scanning
# ---------------------------------------------------------------------------

def scan_multiview_folder(folder: Path | str) -> dict[str, Path]:
    """
    Scan a folder for orientation-suffixed images and return a path dict.

    Filename convention:  <any-prefix>-<orientation>.<ext>
    Recognised orientations: front, left, right, back  (case-insensitive stem match)
    Recognised extensions:   IMAGE_EXTENSIONS

    Example layout:
        inputs/object/
            object-front.png
            object-left.png
            object-right.png
            object-back.png

    Args:
        folder: Directory to scan.

    Returns:
        Dict mapping orientation name → Path, e.g.
        {"front": Path("object-front.png"), "left": ..., ...}

    Raises:
        NotADirectoryError: If ``folder`` does not exist or is not a directory.
        ValueError: If one or more orientations are missing.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Multiview folder not found or not a directory: {folder}")

    found: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem.lower()
        for orient in VIEW_ORDER:
            if stem.endswith(f"-{orient}"):
                if orient not in found:
                    found[orient] = path
                else:
                    logger.warning(
                        "Duplicate '%s' view found — keeping %s, ignoring %s.",
                        orient, found[orient].name, path.name,
                    )

    missing = [o for o in VIEW_ORDER if o not in found]
    if missing:
        raise ValueError(
            f"Multiview folder is missing views {missing}.\n"
            f"  Folder: {folder}\n"
            f"  Found:  {list(found.keys())}\n"
            f"  Expected filenames ending in: "
            + ", ".join(f"-{o}.<ext>" for o in missing)
        )

    logger.info(
        "Multiview folder scanned: %s",
        {o: found[o].name for o in VIEW_ORDER},
    )
    return {o: found[o] for o in VIEW_ORDER}


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
    Run background removal unconditionally (when enabled).

    rembg runs on every input regardless of whether the image already has an
    alpha channel.  Trusting a pre-existing alpha is risky because soft edges
    from JPEG compression, photo-editing apps, or low-quality renders bleed
    background colour into the gray/white composite, which propagates as
    silhouette distortion in shape generation and lighting artefacts in the
    final texture.

    Args:
        image:               Input image (any mode; converted to RGBA internally).
        remove_bg:           Master switch — set False to skip entirely.
        background_remover:  Optional pre-loaded BackgroundRemover instance.
                             If None and removal is needed, one is created lazily.

    Returns:
        RGBA image with background removed (or original RGBA if remove_bg=False).
    """
    image = image.convert("RGBA")

    if not remove_bg:
        return image

    alpha = image.getchannel("A")
    alpha_min, alpha_max = alpha.getextrema()
    logger.debug(
        "Running background removal (alpha_min=%d, alpha_max=%d).",
        alpha_min,
        alpha_max,
    )

    if background_remover is None:
        from hy3dshape.rembg import BackgroundRemover  # type: ignore[import]
        background_remover = BackgroundRemover()

    image = background_remover(image)
    return image


def erode_alpha(image: Image.Image, px: int) -> Image.Image:
    """
    Erode the alpha mask by ``px`` pixels to remove background halo bleed.

    rembg occasionally leaves a 1–3 px fringe of semi-transparent background
    pixels around the foreground silhouette.  A small erosion clips that fringe
    before compositing, preventing background colour from bleeding into the
    gray/white reference images seen by the shape and paint models.

    Args:
        image: RGBA input image.
        px:    Erosion radius in pixels.  0 disables the step (no-op).
               Values of 1–2 are recommended for real photographs; 0 for clean
               CG renders where rembg produces sharp edges already.

    Returns:
        RGBA image with the alpha mask eroded by ``px`` pixels.
    """
    if px <= 0:
        return image

    import cv2  # type: ignore[import]
    import numpy as np

    image = image.convert("RGBA")
    arr = np.array(image)
    kernel = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    arr[..., 3] = cv2.erode(arr[..., 3], kernel, iterations=1)
    return Image.fromarray(arr)


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
    erode_px: int = 0,
    for_shape: bool = False,
    gray: int = 127,
) -> Image.Image:
    """
    Run the full preprocessing chain for one view image.

    Steps:
        1. Remove background (rembg) — always runs when remove_bg=True.
        2. Optionally erode alpha mask to trim halo bleed.
        3. Composite over gray (paint reference) or white (shape input).

    Args:
        image:               Input image.
        remove_bg:           Whether to run background removal.
        background_remover:  Pre-loaded remover instance (reuse across views).
        erode_px:            Pixels to erode from the alpha mask edge (0 = off).
        for_shape:           If True, composite over white (shape conditioning).
                             If False, composite over gray (paint reference).
        gray:                Gray level for paint composite.

    Returns:
        RGB image ready for model input.
    """
    rgba = maybe_remove_background(image, remove_bg=remove_bg, background_remover=background_remover)
    rgba = erode_alpha(rgba, px=erode_px)
    if for_shape:
        return compose_over_white(rgba)
    return compose_over_gray(rgba, gray=gray)


def preprocess_all_views(
    views: dict[str, Image.Image],
    *,
    remove_bg: bool = True,
    erode_px: int = 0,
) -> dict[str, dict[str, Image.Image]]:
    """
    Preprocess all collected view images and return both variants.

    Lazily creates a single BackgroundRemover and reuses it across views
    to avoid loading the ONNX model repeatedly.

    Args:
        views:      Dict of view name → RGBA PIL image (from collect_views).
        remove_bg:  Whether to run background removal.
        erode_px:   Pixels to erode from the alpha mask edge after rembg (0 = off).

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
        rgba = erode_alpha(rgba, px=erode_px)
        result[name] = {
            "rgba": rgba,
            "shape": compose_over_white(rgba),
            "paint": compose_over_gray(rgba),
        }
        logger.debug("Preprocessed view '%s'.", name)

    return result
