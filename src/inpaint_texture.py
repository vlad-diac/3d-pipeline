"""
UV texture inpainting — plan Step N.

Fills uncovered UV texels using a two-pass strategy:

  Pass 1 — vertex-aware inpainting (meshVerticeInpaint C++ extension):
    Propagates colour from covered mesh vertices to uncovered neighbours along
    mesh edges.  Preserves geometric continuity and prevents colour bleeding
    across UV seams.  Requires the DifferentiableRenderer CUDA extension to be
    compiled.  Gracefully skipped if the extension is not available.

  Pass 2 — OpenCV inpainting (cv2.INPAINT_NS or cv2.INPAINT_TELEA):
    Fills any remaining 2-D gaps in UV space using PDE-based region growing.
    "NS" (Navier-Stokes) is the default — it tends to blend more naturally at
    larger uncovered regions.  "TELEA" (Fast Marching) is faster and works
    well for small seam gaps.

The trust mask from bake_texture.bake_multiview_textures is the direct input
to this step.  Texels with mask ≤ 1e-8 are treated as uncovered.

Import note: All GPU-dependent and OpenCV imports are deferred to function
bodies so that this module is importable on macOS without CUDA or cv2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from src.render_multiview import PaintPipeline  # type: ignore[import]

logger = logging.getLogger(__name__)


def inpaint_textures(
    pipeline: "PaintPipeline",
    texture_albedo,
    mask_albedo,
    texture_mr,
    mask_mr,
    *,
    vertex_inpaint: bool = True,
    method: str = "NS",
    save_dir: Optional[Path] = None,
) -> Tuple:
    """
    Two-pass inpainting of UV texture maps.

    Pass 1 (vertex-aware, GPU):
        Uses the compiled meshVerticeInpaint extension to fill uncovered
        texels via mesh-edge propagation.  Produces geometrically coherent
        seam filling.  Skipped automatically if the extension is not compiled.

    Pass 2 (cv2, CPU):
        OpenCV inpainting fills any remaining coverage gaps in 2-D UV space.

    Args:
        pipeline:       Initialized PaintPipeline (ViewProcessor handles the
                        vertex-inpaint pass internally).
        texture_albedo: Baked albedo texture tensor [H, W, 3] in [0, 1].
        mask_albedo:    Coverage mask tensor [H, W, 1] — 1.0 = covered.
        texture_mr:     Baked MR texture tensor [H, W, 3] in [0, 1].
        mask_mr:        Coverage mask tensor for MR.
        vertex_inpaint: If True, attempt vertex-aware pass first.
                        Set False on macOS or when extension is unavailable.
        method:         OpenCV inpainting method: "NS" (Navier-Stokes, default)
                        or "TELEA" (Fast Marching — faster for small gaps).
        save_dir:       Optional directory for --save-intermediates output.

    Returns:
        (refined_albedo, refined_mr) — tensors in the same format as inputs.

    Raises:
        ValueError: If method is not "NS" or "TELEA".
    """
    import torch  # noqa: F401

    if method not in {"NS", "TELEA"}:
        raise ValueError(
            f"inpaint method must be 'NS' or 'TELEA', got: {method!r}"
        )

    mask_albedo_u8 = _mask_to_uint8(mask_albedo)
    mask_mr_u8 = _mask_to_uint8(mask_mr)

    logger.info(
        "Inpainting textures — vertex_inpaint=%s  method=%s.", vertex_inpaint, method
    )

    with torch.inference_mode():
        refined_albedo = pipeline.view_processor.texture_inpaint(
            texture_albedo, mask_albedo_u8, vertex_inpaint, method
        )
        refined_mr = pipeline.view_processor.texture_inpaint(
            texture_mr, mask_mr_u8, vertex_inpaint, method
        )

    logger.info("Inpainting complete.")

    if save_dir is not None:
        _save_inpaint_outputs(save_dir, refined_albedo, refined_mr)

    return refined_albedo, refined_mr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_to_uint8(mask) -> np.ndarray:
    """
    Convert a coverage mask tensor / array to a uint8 numpy array (0 or 255).

    The mask from bake_texture is a float tensor [H, W, 1] where values ≤ 1e-8
    mean "uncovered".  We convert to a binary uint8 image (0 = hole, 255 = covered)
    for OpenCV compatibility.
    """
    if hasattr(mask, "detach"):
        arr = mask.squeeze(-1).detach().float().cpu().numpy()
    elif hasattr(mask, "squeeze"):
        arr = np.asarray(mask).squeeze()
    else:
        arr = np.asarray(mask)

    arr = np.squeeze(arr)
    arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _save_inpaint_outputs(
    save_dir: Path,
    refined_albedo,
    refined_mr,
) -> None:
    """Save refined texture tensors as PNG intermediates."""
    from PIL import Image

    save_dir.mkdir(parents=True, exist_ok=True)

    def _tensor_to_pil(t) -> Image.Image:
        if hasattr(t, "detach"):
            arr = t.detach().float().cpu().numpy()
        else:
            arr = np.asarray(t)
        arr = np.clip(arr, 0.0, 1.0)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            return Image.fromarray((arr * 255).astype(np.uint8), mode="RGBA")
        elif arr.ndim == 2:
            return Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        else:
            raise ValueError(f"Cannot convert tensor of shape {arr.shape}.")

    try:
        _tensor_to_pil(refined_albedo).save(str(save_dir / "refined_albedo.png"))
        _tensor_to_pil(refined_mr).save(str(save_dir / "refined_mr.png"))
        logger.info("Inpaint intermediates saved to %s.", save_dir)
    except Exception as exc:
        logger.warning("Could not save inpaint intermediates: %s", exc)
