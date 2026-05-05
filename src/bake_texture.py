"""
UV texture baking — plan Step M.

Back-projects multiview colour images (albedo + metallic-roughness) into the
UV texture atlas using cosine-weighted blending.

Algorithm per UV texel:
  1. Determine which camera views can see this texel (visibility test via
     the compiled custom_rasterizer CUDA extension).
  2. For each visible view, sample the colour from the rendered view image at
     the projected screen position.
  3. Compute blend weight  =  base_view_weight × cos(θ)^bake_exp
     where θ is the angle between the surface normal and the camera ray.
  4. Accumulate weighted colour sums and a coverage mask.
  5. Divide by total weight for each covered texel; record coverage in a
     trust mask (1.0 = covered, 0.0 = needs inpainting).

The trust mask from this step is consumed directly by inpaint_texture.py.

Import note: All GPU-dependent imports are deferred to function bodies so
that this module is importable on macOS without CUDA.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from PIL import Image

if TYPE_CHECKING:
    from src.render_multiview import PaintPipeline  # type: ignore[import]

logger = logging.getLogger(__name__)


def bake_multiview_textures(
    pipeline: "PaintPipeline",
    albedo_views: List[Image.Image],
    mr_views: List[Image.Image],
    cfg,
    *,
    save_dir: Optional[Path] = None,
) -> Tuple:
    """
    Bake albedo and metallic-roughness views into UV texture maps.

    Both albedo and MR share the same camera configuration and bake weights,
    so they are baked in the same pass (same visibility / projection for each
    texel).  The returned masks are identical in coverage but returned
    separately so downstream inpainting can treat them independently.

    Args:
        pipeline:     Initialized PaintPipeline with UV-unwrapped mesh loaded.
        albedo_views: List of upscaled albedo PIL images, one per camera view,
                      in the same order as cfg.camera.azimuths.
        mr_views:     Corresponding metallic-roughness images.
        cfg:          PipelineConfig — uses cfg.render_size, cfg.camera.
        save_dir:     Optional directory.  When set, saves baked textures and
                      masks as PNG files (for --save-intermediates).

    Returns:
        (texture_albedo, mask_albedo, texture_mr, mask_mr)
          texture_*  — torch.Tensor [H, W, 3] in [0, 1] (or framework native)
          mask_*     — torch.Tensor [H, W, 1] — 1.0 covered, 0.0 uncovered

    Raises:
        RuntimeError: If the ViewProcessor raises during bake (e.g. extension
                      not compiled on macOS).
    """
    import torch  # noqa: F401 (imported for inference_mode)

    render_size = cfg.render_size
    n_albedo = len(albedo_views)
    n_mr = len(mr_views)

    logger.info(
        "Baking %d albedo views + %d MR views → texture_size=%d  render_size=%d.",
        n_albedo, n_mr, cfg.texture_size, render_size,
    )

    # Ensure all views are at render_size before baking for maximum precision.
    albedo_resized = [
        img.resize((render_size, render_size), Image.Resampling.LANCZOS)
        for img in albedo_views
    ]
    mr_resized = [
        img.resize((render_size, render_size), Image.Resampling.LANCZOS)
        for img in mr_views
    ]

    with torch.inference_mode():
        texture_albedo, mask_albedo = pipeline.view_processor.bake_from_multiview(
            albedo_resized,
            cfg.camera.elevations,
            cfg.camera.azimuths,
            cfg.camera.weights,
        )

        texture_mr, mask_mr = pipeline.view_processor.bake_from_multiview(
            mr_resized,
            cfg.camera.elevations,
            cfg.camera.azimuths,
            cfg.camera.weights,
        )

    # Coverage statistics
    try:
        import torch as _torch
        albedo_arr = mask_albedo.squeeze(-1) if hasattr(mask_albedo, "squeeze") else mask_albedo
        covered_px = int((albedo_arr > 1e-8).sum().item())
        total_px = int(albedo_arr.numel())
        pct = 100.0 * covered_px / total_px if total_px > 0 else 0.0
        logger.info(
            "Bake complete: %d / %d texels covered (%.1f%%) — %.1f%% need inpainting.",
            covered_px, total_px, pct, 100.0 - pct,
        )
    except Exception:
        pass

    if save_dir is not None:
        _save_bake_outputs(
            save_dir, texture_albedo, mask_albedo, texture_mr, mask_mr
        )

    return texture_albedo, mask_albedo, texture_mr, mask_mr


# ---------------------------------------------------------------------------
# Internal: save intermediate bake outputs
# ---------------------------------------------------------------------------

def _save_bake_outputs(
    save_dir: Path,
    texture_albedo,
    mask_albedo,
    texture_mr,
    mask_mr,
) -> None:
    """Save baked textures and coverage masks as PNG files."""
    save_dir.mkdir(parents=True, exist_ok=True)

    def _tensor_to_pil(t, channels: int = 3) -> Image.Image:
        """Convert a [H, W, C] or [H, W] tensor/array to a PIL image."""
        import numpy as np

        if hasattr(t, "detach"):
            arr = t.detach().float().cpu().numpy()
        else:
            arr = np.asarray(t)

        arr = np.clip(arr, 0.0, 1.0)

        if arr.ndim == 2:
            return Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            return Image.fromarray((arr[:, :, 0] * 255).astype(np.uint8), mode="L")
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            return Image.fromarray((arr * 255).astype(np.uint8), mode="RGBA")
        else:
            raise ValueError(f"Cannot convert array of shape {arr.shape} to PIL image.")

    try:
        _tensor_to_pil(texture_albedo).save(str(save_dir / "baked_albedo.png"))
        _tensor_to_pil(mask_albedo).save(str(save_dir / "baked_albedo_mask.png"))
        _tensor_to_pil(texture_mr).save(str(save_dir / "baked_mr.png"))
        _tensor_to_pil(mask_mr).save(str(save_dir / "baked_mr_mask.png"))
        logger.info("Bake intermediates saved to %s.", save_dir)
    except Exception as exc:
        logger.warning("Could not save bake intermediates: %s", exc)
