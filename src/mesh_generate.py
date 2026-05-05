"""
Mesh generation — plan Steps C and D.

Step C: load the shape pipeline (Hunyuan3DDiTFlowMatchingPipeline).
Step D: run shape diffusion and return a trimesh.Trimesh.

Two geometry paths are supported (controlled by cfg.use_multiview_shape):
  Path A — single-image  (v2.1 DiT, tencent/Hunyuan3D-2.1)
  Path B — four-view     (v2.0 2mv DiT, tencent/Hunyuan3D-2mv)

Both converge to the same downstream texture pipeline.

Loading strategy (load_shape_pipeline_auto):
  1. If the local models/ directory contains the checkpoint, load from disk
     (avoids network access on RunPod after download_models.py has run).
  2. Otherwise, fall back to from_pretrained with the HuggingFace repo ID
     (uses the HF cache if previously downloaded, otherwise downloads).

Key decisions from the plan:
  - Use output_type="trimesh"  — stable public API; avoids the fragile latent split.
  - Use torch.inference_mode() — more efficient than no_grad.
  - Use torch.Generator(device=...) — device-aware seeding.
  - Delete pipeline + empty_cache() immediately after generation — shape uses ~10 GB VRAM.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Step C: Load shape pipeline
# ---------------------------------------------------------------------------

def load_shape_pipeline(cfg):
    """
    Load the shape pipeline from HuggingFace (or HF cache).

    Args:
        cfg: PipelineConfig

    Returns:
        Hunyuan3DDiTFlowMatchingPipeline ready for inference.
    """
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import]

    if cfg.use_multiview_shape:
        logger.info(
            "Loading four-view shape pipeline (tencent/Hunyuan3D-2mv / %s).",
            cfg.shape_model_subfolder,
        )
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mv",
            subfolder=cfg.shape_model_subfolder,
            device=cfg.device,
            dtype=torch.float16,
        )
    else:
        logger.info("Loading single-image shape pipeline (tencent/Hunyuan3D-2.1).")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2.1",
            subfolder="hunyuan3d-dit-v2-1",
            device=cfg.device,
            dtype=torch.float16,
        )

    return pipeline


def load_shape_pipeline_local(cfg):
    """
    Load the shape pipeline from a local checkpoint (offline / custom weights).

    Expects:
        models/hunyuan3d-dit-v2-1/model.fp16.ckpt
        models/hunyuan3d-dit-v2-1/config.yaml

    Raises:
        FileNotFoundError: If either file is missing.
    """
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import]

    subfolder = "hunyuan3d-dit-v2-mv" if cfg.use_multiview_shape else "hunyuan3d-dit-v2-1"
    model_dir = cfg.models_dir / subfolder / subfolder
    ckpt_path = model_dir / "model.fp16.ckpt"
    config_path = model_dir / "config.yaml"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Shape checkpoint not found: {ckpt_path}\n"
            "Run:  python scripts/download_models.py  to download model weights."
        )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Shape config not found: {config_path}\n"
            "Run:  python scripts/download_models.py  to download model weights."
        )

    logger.info("Loading shape pipeline from local checkpoint: %s", ckpt_path)
    return Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=str(ckpt_path),
        config_path=str(config_path),
        device=cfg.device,
        dtype=torch.float16,
        use_safetensors=False,
    )


def load_shape_pipeline_auto(cfg):
    """
    Smart loader: try local checkpoint first, fall back to HuggingFace.

    This is the recommended entry point for normal pipeline usage:
    - On RunPod after download_models.py: uses local weights (no network).
    - On a fresh machine without weights: downloads from HuggingFace.

    Args:
        cfg: PipelineConfig

    Returns:
        Hunyuan3DDiTFlowMatchingPipeline ready for inference.
    """
    subfolder = "hunyuan3d-dit-v2-mv" if cfg.use_multiview_shape else "hunyuan3d-dit-v2-1"
    local_ckpt = cfg.models_dir / subfolder / subfolder / "model.fp16.ckpt"

    if local_ckpt.exists():
        logger.info("Local checkpoint found — loading from disk.")
        return load_shape_pipeline_local(cfg)

    logger.info("Local checkpoint not found — loading from HuggingFace.")
    return load_shape_pipeline(cfg)


# ---------------------------------------------------------------------------
# Step D: Generate mesh
# ---------------------------------------------------------------------------

def generate_mesh(pipeline, views: dict, cfg):
    """
    Run shape diffusion and return a trimesh.Trimesh.

    Args:
        pipeline: Loaded Hunyuan3DDiTFlowMatchingPipeline.
        views:    Dict[str, PIL.Image] from preprocess.collect_views.
                  Single-image mode: {"front": img}
                  Four-view mode:    {"front": img, "left": img, "right": img, "back": img}
        cfg:      PipelineConfig

    Returns:
        trimesh.Trimesh — raw mesh directly from the shape DiT.

    Notes:
        - output_type="trimesh" is the stable public API (avoids unstable latent split).
        - The pipeline and its GPU memory are freed immediately after mesh extraction.
          (Shape DiT uses ~10 GB VRAM; must be freed before paint pipeline loads.)
    """
    import torch
    import trimesh

    seed = int(cfg.seed) % (2 ** 32)
    generator = torch.Generator(device=cfg.device).manual_seed(seed)

    if len(views) == 1:
        image_input = list(views.values())[0]
        logger.info("Generating mesh from single view (seed=%d, steps=%d).", seed, cfg.shape_steps)
    else:
        # MVImageProcessorV2 (hy3dgen/shapegen/preprocessors.py) expects a dict
        # keyed by orientation name {"front", "left", "right", "back"}.
        # It uses view2idx = {'front':0,'left':1,'back':2,'right':3} to sort
        # internally, so ordering in the dict does not matter — just the keys.
        image_input = views
        logger.info(
            "Generating mesh from %d views %s (seed=%d, steps=%d).",
            len(views),
            list(views.keys()),
            seed,
            cfg.shape_steps,
        )

    with torch.inference_mode():
        result = pipeline(
            image=image_input,
            num_inference_steps=cfg.shape_steps,
            guidance_scale=cfg.shape_guidance_scale,
            generator=generator,
            octree_resolution=cfg.octree_resolution,
            output_type="trimesh",
        )

    mesh = result[0] if isinstance(result, (list, tuple)) else result

    if not isinstance(mesh, trimesh.Trimesh):
        # Some pipeline versions wrap the mesh in a container — unwrap it.
        if hasattr(mesh, "mesh_v") and hasattr(mesh, "mesh_f"):
            import numpy as np
            faces = mesh.mesh_f
            if faces.ndim == 2 and faces.shape[1] == 3:
                faces = faces[:, ::-1]  # flip winding order
            mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.mesh_v),
                faces=np.asarray(faces),
                process=False,
            )
        else:
            raise TypeError(
                f"Unexpected pipeline output type: {type(mesh)}. "
                "Expected trimesh.Trimesh or an object with .mesh_v / .mesh_f."
            )

    logger.info(
        "Mesh generated: %d vertices, %d faces.",
        len(mesh.vertices),
        len(mesh.faces),
    )

    _free_pipeline(pipeline)
    return mesh


def _free_pipeline(pipeline) -> None:
    """Delete pipeline and release GPU memory."""
    import torch
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("Shape pipeline freed from GPU memory.")


# ---------------------------------------------------------------------------
# Advanced: explicit latent split (custom / research usage only)
# ---------------------------------------------------------------------------

def generate_mesh_via_latent_split(pipeline, vae, image, cfg):
    """
    ADVANCED: Separate latent generation from VAE decode.

    Only use this if you need to cache or modify latents between steps.
    WARNING: output_type="latent" is not guaranteed stable across releases.
             Prefer generate_mesh() for normal pipeline usage.

    Args:
        pipeline: Shape DiT pipeline.
        vae:      Separately loaded VAE (Hunyuan3D-VAE v2.1).
        image:    Single PIL.Image (front view).
        cfg:      PipelineConfig.

    Returns:
        trimesh.Trimesh decoded via the explicit VAE.
    """
    import torch
    import numpy as np
    import trimesh

    with torch.inference_mode():
        try:
            latents = pipeline(
                image=image,
                num_inference_steps=cfg.shape_steps,
                guidance_scale=cfg.shape_guidance_scale,
                generator=torch.Generator(device=cfg.device).manual_seed(cfg.seed),
                output_type="latent",
            )
        except TypeError as exc:
            raise RuntimeError(
                "This pipeline version does not support output_type='latent'. "
                "Use generate_mesh() with output_type='trimesh' instead."
            ) from exc

        latents = vae.decode(latents)
        outputs = vae.latents2mesh(
            latents,
            output_type="trimesh",
            bounds=1.01,
            mc_level=0.0,
            num_chunks=8000,
            octree_resolution=cfg.octree_resolution,
            mc_algo="mc",
            enable_pbar=True,
        )[0]

    faces = outputs.mesh_f
    if faces.ndim == 2 and faces.shape[1] == 3:
        faces = faces[:, ::-1]

    mesh = trimesh.Trimesh(
        vertices=np.asarray(outputs.mesh_v),
        faces=np.asarray(faces),
        process=False,
    )

    logger.info(
        "Mesh generated via latent split: %d vertices, %d faces.",
        len(mesh.vertices),
        len(mesh.faces),
    )

    _free_pipeline(pipeline)
    return mesh
