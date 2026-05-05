"""
Canonical multiview generation using MV-Adapter i2mv SDXL.

Pipeline stage: ``remove_background → [canonical_multiview] → preprocess``

This module takes a *clean RGBA* anchor image (already background-removed and
centred by the ``remove_background`` stage) and uses MV-Adapter's image-to-
multiview SDXL pipeline to synthesise four canonical views:

    front (az=0°)  · right (az=90°)  · back (az=180°)  · left (az=270°)

Optionally adds depth (DPT-hybrid-midas) and canny structural conditioning via
a local fork of the MV-Adapter i2mv pipeline that exposes the ``controlnet``
branch from the t2mv pipeline.

No background removal happens here — the caller is expected to supply clean RGBA.

References
----------
* MV-Adapter repo   : https://github.com/huanngzh/MV-Adapter
* MV-Adapter weights: https://huggingface.co/huanngzh/mv-adapter
* Research doc      : docs/3d/canonical-multiview-generation.md

Dependencies (all lazy-imported to allow CPU imports)
------------------------------------------------------
* mvadapter          — MV-Adapter Python package (third_party/MV-Adapter)
* diffusers          — SDXL base + VAE + optional ControlNet
* transformers       — DPT depth estimator
* torch / torchvision
* cv2
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PIL import Image

if TYPE_CHECKING:
    from src.config import CanonicalMultiviewConfig

logger = logging.getLogger(__name__)

# Canonical four-view schedule: (name, azimuth_deg)
VIEWS: list[tuple[str, int]] = [
    ("front",  0),
    ("right",  90),
    ("back",   180),
    ("left",   270),
]
VIEW_NAMES = [name for name, _ in VIEWS]
AZIMUTHS   = [az   for _, az   in VIEWS]


def _add_mvadapter_to_path() -> None:
    """Ensure third_party/MV-Adapter is on sys.path."""
    project_root = Path(__file__).resolve().parents[1]
    mv_adapter_dir = project_root / "third_party" / "MV-Adapter"
    if mv_adapter_dir.exists() and str(mv_adapter_dir) not in sys.path:
        sys.path.insert(0, str(mv_adapter_dir))
        logger.debug("Added %s to sys.path.", mv_adapter_dir)


# ---------------------------------------------------------------------------
# Depth and edge conditioning
# ---------------------------------------------------------------------------

def build_depth_map(clean_rgba: Image.Image, device: str, gen_size: int = 768) -> Image.Image:
    """
    Estimate depth from a clean RGBA image using DPT-hybrid-midas.

    Runs the estimator at 1024 px for structure fidelity, then resizes the
    depth map to ``gen_size`` for ControlNet input.

    Args:
        clean_rgba: Pre-cleaned RGBA image (from remove_background stage).
        device:     PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
        gen_size:   Output size in pixels (should match ``cfg.gen_size``).

    Returns:
        RGB depth map at ``gen_size × gen_size`` suitable for ControlNet input.
    """
    import numpy as np
    import torch
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation

    logger.info("Building depth map (DPT-hybrid-midas) ...")
    rgb_img = clean_rgba.convert("RGB")

    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
    depth_model.eval()

    pixel_values = feature_extractor(images=rgb_img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", enabled=device.startswith("cuda")
    ):
        depth_out = depth_model(pixel_values).predicted_depth

    # Interpolate to 1024 then normalise to [0, 1]
    depth = torch.nn.functional.interpolate(
        depth_out.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    d_min = torch.amin(depth, dim=[1, 2, 3], keepdim=True)
    d_max = torch.amax(depth, dim=[1, 2, 3], keepdim=True)
    depth = (depth - d_min) / (d_max - d_min + 1e-8)

    # Convert to uint8 RGB
    arr = (
        torch.cat([depth] * 3, dim=1)[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    depth_img = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))

    # Resize to gen_size for ControlNet input
    depth_img = depth_img.resize((gen_size, gen_size), Image.Resampling.LANCZOS)

    del depth_model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return depth_img


def build_canny_map(clean_rgba: Image.Image, gen_size: int = 768) -> Image.Image:
    """
    Extract canny edges from a clean RGBA image.

    Uses a weak threshold pair so that the result acts as a light structural
    guide rather than over-constraining the diffusion output.

    Args:
        clean_rgba: Pre-cleaned RGBA image.
        gen_size:   Output size in pixels.

    Returns:
        RGB canny edge map at ``gen_size × gen_size``.
    """
    import cv2
    import numpy as np

    rgb = np.array(clean_rgba.convert("RGB"))
    edges = cv2.Canny(rgb, 100, 200)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    img = Image.fromarray(edges_rgb)
    return img.resize((gen_size, gen_size), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Plücker camera controls (MV-Adapter camera conditioning)
# ---------------------------------------------------------------------------

def build_plucker_controls(gen_size: int, device: str):  # -> torch.Tensor
    """
    Build the Plücker camera-embedding tensor for the four canonical views.

    Uses the orthographic camera model from MV-Adapter's utilities.

    Args:
        gen_size: Spatial size matching the generation resolution.
        device:   PyTorch device string.

    Returns:
        Float tensor ``(4, 6, gen_size, gen_size)`` in [0, 1].

    Raises:
        ImportError: If MV-Adapter is not found in ``third_party/MV-Adapter``.
    """
    _add_mvadapter_to_path()
    from mvadapter.utils.mesh_utils import get_orthogonal_camera  # type: ignore[import]
    from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho  # type: ignore[import]

    cameras = get_orthogonal_camera(
        elevation_deg=[0] * len(VIEWS),
        distance=[1.8] * len(VIEWS),
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        # MV-Adapter's convention: azimuth offset by -90°
        azimuth_deg=[az - 90 for az in AZIMUTHS],
        device=device,
    )
    plucker = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * len(VIEWS), gen_size
    )
    return ((plucker + 1.0) / 2.0).clamp(0, 1)


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_mvadapter_pipe(cfg: "CanonicalMultiviewConfig", device: str):
    """
    Load the MV-Adapter i2mv SDXL pipeline, optionally with ControlNets.

    If ``cfg.use_depth`` or ``cfg.use_canny`` is True, the function attempts
    to use the local ControlNet fork
    ``mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl_controlnet``.
    If the fork is not present the pipeline falls back to the standard i2mv
    SDXL pipeline without ControlNet support and logs a warning.

    Args:
        cfg:    ``CanonicalMultiviewConfig`` instance with model paths.
        device: PyTorch device string.

    Returns:
        Loaded pipeline instance, moved to ``device``.

    Raises:
        ImportError: If neither mvadapter nor diffusers is available.
    """
    import torch
    from diffusers import AutoencoderKL, ControlNetModel  # type: ignore[import]

    _add_mvadapter_to_path()

    vae = AutoencoderKL.from_pretrained(
        cfg.vae_model, torch_dtype=torch.float16
    ).to(device)

    want_controlnet = cfg.use_depth or cfg.use_canny
    controlnets = []

    if want_controlnet:
        if cfg.use_depth:
            logger.info("Loading depth ControlNet ...")
            controlnets.append(
                ControlNetModel.from_pretrained(
                    "diffusers/controlnet-depth-sdxl-1.0",
                    variant="fp16",
                    use_safetensors=True,
                    torch_dtype=torch.float16,
                ).to(device)
            )
        if cfg.use_canny:
            logger.info("Loading canny ControlNet ...")
            controlnets.append(
                ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0",
                    torch_dtype=torch.float16,
                ).to(device)
            )

    # Try local ControlNet fork first; fall back to standard i2mv if absent.
    use_cn_fork = want_controlnet and bool(controlnets)
    if use_cn_fork:
        try:
            from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl_controlnet import (  # type: ignore[import]
                MVAdapterI2MVSDXLControlNetPipeline,
            )
            PipelineClass = MVAdapterI2MVSDXLControlNetPipeline
            logger.info("Using MVAdapterI2MVSDXLControlNetPipeline (local fork).")
        except ImportError:
            logger.warning(
                "Local ControlNet fork (pipeline_mvadapter_i2mv_sdxl_controlnet) "
                "not found — falling back to standard i2mv pipeline without ControlNet. "
                "See docs/3d/canonical-multiview-generation.md §ControlNet integration."
            )
            from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import (  # type: ignore[import]
                MVAdapterI2MVSDXLPipeline,
            )
            PipelineClass = MVAdapterI2MVSDXLPipeline
            controlnets = []
    else:
        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import (  # type: ignore[import]
            MVAdapterI2MVSDXLPipeline,
        )
        PipelineClass = MVAdapterI2MVSDXLPipeline

    logger.info("Loading SDXL base + MV-Adapter weights ...")
    pipe_kwargs: dict = {
        "vae": vae,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if controlnets:
        pipe_kwargs["controlnet"] = controlnets if len(controlnets) > 1 else controlnets[0]

    pipe = PipelineClass.from_pretrained(cfg.base_model, **pipe_kwargs)
    pipe.to(device)  # move SDXL weights to GPU before loading the adapter on top
    pipe.init_custom_adapter(num_views=len(VIEWS))
    pipe.load_custom_adapter(cfg.adapter_repo, weight_name=cfg.adapter_weight)
    pipe.enable_vae_slicing()

    if hasattr(pipe, "cond_encoder"):
        pipe.cond_encoder.to(device=device, dtype=torch.float16)

    return pipe


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def _anchor_rgb(clean_rgba: Image.Image, gen_size: int) -> Image.Image:
    """
    Prepare the MV-Adapter anchor image: composite RGBA over gray, resize to gen_size.

    The official MV-Adapter script composites the anchor over gray 0.5 before
    feeding it to the reference encoder.
    """
    import numpy as np

    arr = np.array(clean_rgba.convert("RGBA")).astype(np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    rgb = arr[:, :, :3] * alpha + 0.5 * (1.0 - alpha)
    rgb_img = Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")
    return rgb_img.resize((gen_size, gen_size), Image.Resampling.LANCZOS)


def generate_canonical_views(
    clean_rgba: Image.Image,
    cfg: "CanonicalMultiviewConfig",
    save_dir: Optional[Path] = None,
) -> dict[str, Image.Image]:
    """
    Generate canonical ``front/right/back/left`` views from a single clean RGBA.

    The input must already be background-removed and centred (produced by the
    ``remove_background`` pipeline stage).  This function does **no** background
    removal of its own.

    Args:
        clean_rgba: Clean RGBA anchor image (no background).
        cfg:        ``CanonicalMultiviewConfig`` with all generation settings.
        save_dir:   Optional directory to save intermediate and output images.

    Returns:
        Dict ``{"front": PIL.Image, "right": PIL.Image, "back": PIL.Image, "left": PIL.Image}``
        where each value is an RGBA image (opaque, full-alpha).

    Raises:
        ImportError: If MV-Adapter or diffusers is not installed.
        RuntimeError: If generation fails.
    """
    import torch

    device = "cuda" if _cuda_available() else "cpu"
    gen_size = cfg.gen_size

    logger.info(
        "Generating canonical views (MV-Adapter)  steps=%d  guidance=%.1f  "
        "depth=%s  canny=%s  device=%s",
        cfg.steps, cfg.guidance_scale, cfg.use_depth, cfg.use_canny, device,
    )

    # Prepare anchor image composited over gray (MV-Adapter convention)
    anchor_rgb = _anchor_rgb(clean_rgba, gen_size)

    # Build structural control images
    depth_img: Optional[Image.Image] = None
    canny_img: Optional[Image.Image] = None

    if cfg.use_depth:
        with _timer("build_depth_map"):
            depth_img = build_depth_map(clean_rgba, device=device, gen_size=gen_size)
    if cfg.use_canny:
        with _timer("build_canny_map"):
            canny_img = build_canny_map(clean_rgba, gen_size=gen_size)

    # Build Plücker camera controls
    with _timer("build_plucker_controls"):
        plucker_controls = build_plucker_controls(gen_size, device)

    # Load pipeline
    with _timer("load_mvadapter_pipe"):
        pipe = load_mvadapter_pipe(cfg, device)

    # Assemble ControlNet inputs
    controlnet_images: list[Image.Image] = []
    controlnet_scales: list[float] = []
    if depth_img is not None:
        controlnet_images.append(depth_img)
        controlnet_scales.append(cfg.depth_scale)
    if canny_img is not None:
        controlnet_images.append(canny_img)
        controlnet_scales.append(cfg.canny_scale)

    # Run generation
    gen_kwargs: dict = dict(
        prompt=cfg.prompt,
        height=gen_size,
        width=gen_size,
        num_inference_steps=cfg.steps,
        guidance_scale=cfg.guidance_scale,
        num_images_per_prompt=len(VIEWS),
        control_image=plucker_controls,
        control_conditioning_scale=1.0,
        reference_image=anchor_rgb,
        reference_conditioning_scale=cfg.reference_conditioning_scale,
        negative_prompt=cfg.negative_prompt,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    if controlnet_images:
        gen_kwargs["controlnet_image"] = controlnet_images
        gen_kwargs["controlnet_conditioning_scale"] = controlnet_scales

    with _timer("pipe inference"):
        result_images = pipe(**gen_kwargs).images

    # Free GPU memory
    del pipe
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Map to canonical dict — output as RGBA (opaque)
    canonical: dict[str, Image.Image] = {}
    for (name, _), img in zip(VIEWS, result_images):
        canonical[name] = img.convert("RGBA")

    # Optionally save outputs
    if save_dir is not None:
        save_dir = Path(save_dir)
        (save_dir / "preproc").mkdir(parents=True, exist_ok=True)
        (save_dir / "outputs").mkdir(parents=True, exist_ok=True)

        anchor_rgb.save(str(save_dir / "preproc" / "anchor.png"))
        if depth_img:
            depth_img.save(str(save_dir / "preproc" / "depth.png"))
        if canny_img:
            canny_img.save(str(save_dir / "preproc" / "canny.png"))

        for name, img in canonical.items():
            img.save(str(save_dir / "outputs" / f"{name}.png"))

        _save_grid(canonical, save_dir / "grid.png")
        _save_manifest(canonical, cfg, save_dir)

    logger.info("Canonical views generated: %s", list(canonical.keys()))
    return canonical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class _timer:
    """Minimal context manager that logs elapsed time."""
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        import time
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        import time
        logger.info("%s  %.1f s", self.name, time.perf_counter() - self._t0)


def _save_grid(canonical: dict[str, Image.Image], path: Path) -> None:
    """Save a 1×4 grid of the four canonical views."""
    try:
        images = [canonical[n].convert("RGB") for n in VIEW_NAMES if n in canonical]
        if not images:
            return
        w, h = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, img in enumerate(images):
            grid.paste(img, (i * w, 0))
        grid.save(str(path))
        logger.info("Saved canonical grid → %s", path)
    except Exception as exc:
        logger.warning("Could not save grid: %s", exc)


def _save_manifest(
    canonical: dict[str, Image.Image],
    cfg: "CanonicalMultiviewConfig",
    run_dir: Path,
) -> None:
    """Write a manifest.json describing the generation run."""
    manifest = {
        "prompt": cfg.prompt,
        "generation": {
            "size":                         cfg.gen_size,
            "steps":                        cfg.steps,
            "guidance_scale":               cfg.guidance_scale,
            "reference_conditioning_scale": cfg.reference_conditioning_scale,
        },
        "controls": {
            "depth":        "preproc/depth.png" if cfg.use_depth else None,
            "canny":        "preproc/canny.png" if cfg.use_canny else None,
            "depth_scale":  cfg.depth_scale,
            "canny_scale":  cfg.canny_scale,
        },
        "views": [
            {"name": name, "azimuth_deg": az, "path": f"outputs/{name}.png"}
            for name, az in VIEWS
        ],
        "hunyuan_input": {name: f"outputs/{name}.png" for name, _ in VIEWS},
    }
    out = run_dir / "manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest → %s", out)
