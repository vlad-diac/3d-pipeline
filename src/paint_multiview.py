"""
Multiview texture diffusion — delight, paint, upscale.

Covers plan Steps J, K, and L:
  J — Delight reference image (optional Light_Shadow_Remover).
  K — Run multiview PBR diffusion (HunyuanPaintPipeline).
  L — Upscale generated views to render_size (Lanczos or RealESRGAN).

Import note: All GPU-dependent and hy3dpaint imports are deferred to
function/class bodies.  src/__init__.py has already inserted third_party
paths into sys.path.  This module is importable on macOS without CUDA.

Memory note: MultiviewDiffusionNet loads ~21 GB of weights.
Always load it after the shape pipeline has been deleted and
torch.cuda.empty_cache() has been called (see mesh_generate._free_pipeline).

Call pipeline.unload() when done to free VRAM before the bake stage.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step J: Delight reference image (optional)
# ---------------------------------------------------------------------------

def delight_reference(front_rgba: Image.Image, cfg) -> Image.Image:
    """
    Remove directional lighting from the reference image before paint diffusion.

    Process:
      1. Composite front view onto neutral gray (reduces lighting bias).
      2. Run Light_Shadow_Remover for flat-lit appearance reference.

    If the delight model checkpoint is not found, or the import fails, the
    function returns the plain gray composite.  The step is optional but
    improves texture quality on real-world photographs by preventing baked-in
    highlights and shadows from propagating into the UV texture.

    Args:
        front_rgba: Front-view RGBA image (any size).
        cfg:        PipelineConfig — uses cfg.delight_model_dir, cfg.device.

    Returns:
        RGB PIL.Image composited over gray, optionally delighted.
    """
    from src.preprocess import compose_over_gray  # type: ignore[import]

    reference = compose_over_gray(front_rgba, gray=127)

    if not cfg.delight_model_dir.exists():
        logger.info(
            "Delight model not found at %s — skipping (returning gray composite).",
            cfg.delight_model_dir,
        )
        return reference

    try:
        from hy3dgen.texgen.utils.dehighlight_utils import Light_Shadow_Remover  # type: ignore[import]

        delight_cfg = SimpleNamespace(
            device=cfg.device,
            light_remover_ckpt_path=str(cfg.delight_model_dir),
        )
        delight = Light_Shadow_Remover(delight_cfg)
        delighted = delight(reference)
        logger.info("Delight step complete.")
        return delighted

    except ImportError:
        logger.warning(
            "hy3dgen.texgen.utils.dehighlight_utils not importable — "
            "skipping delight (expected if Hunyuan3D-2 is not cloned)."
        )
        return reference

    except Exception as exc:
        logger.warning(
            "Delight step raised %s: %s — falling back to plain gray composite.",
            type(exc).__name__, exc,
        )
        return reference


# ---------------------------------------------------------------------------
# Step K: Multiview paint diffusion
# ---------------------------------------------------------------------------

class MultiviewDiffusionNet:
    """
    Multiview PBR texture diffusion model (HunyuanPaintPipeline).

    Loads the pipeline from the local models directory (offline after
    download_models.py has run), conditionally loads DINOv2 for image
    feature conditioning, and generates 6 albedo + 6 metallic-roughness
    views from the conditioning maps produced in render_multiview.py.

    Scheduler: UniPCMultistepScheduler with trailing timestep spacing
    (upstream default — good quality at 15 steps).

    VAE optimizations: slicing + tiling enabled to reduce peak VRAM.

    Args:
        cfg: PipelineConfig.  Required paths:
               cfg.models_dir / "hunyuan3d-paintpbr-v2-1"  — pipeline weights
               cfg.models_dir / "dinov2-giant"              — DINOv2 (if needed)
               cfg.third_party_dir / "Hunyuan3D-2.1" / "hy3dpaint" / "cfgs"
                   / "hunyuan-paint-pbr.yaml"               — mode + view_size
    """

    def __init__(self, cfg) -> None:
        import torch
        from diffusers import DiffusionPipeline, UniPCMultistepScheduler  # type: ignore[import]
        from omegaconf import OmegaConf  # type: ignore[import]

        self.device = cfg.device

        # ---- validate model directory ----------------------------------------
        model_dir = cfg.models_dir / "hunyuan3d-paintpbr-v2-1"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Paint model not found: {model_dir}\n"
                "Run:  python scripts/download_models.py  to download model weights."
            )

        # ---- detect PBR mode and view_size from YAML config ------------------
        paint_cfg_path = (
            cfg.third_party_dir
            / "Hunyuan3D-2.1"
            / "hy3dpaint"
            / "cfgs"
            / "hunyuan-paint-pbr.yaml"
        )
        if paint_cfg_path.exists():
            paint_cfg = OmegaConf.load(str(paint_cfg_path))
            self.mode: str = paint_cfg.model.params.stable_diffusion_config.custom_pipeline[2:]
            cfg_view_size: int = int(paint_cfg.model.params.get("view_size", 320))
        else:
            logger.warning(
                "Paint config YAML not found at %s — defaulting mode='pbr', view_size=320.",
                paint_cfg_path,
            )
            self.mode = "pbr"
            cfg_view_size = 320

        # ---- custom_pipeline points to the local hunyuanpaintpbr/ directory --
        # DiffusionPipeline.from_pretrained needs the absolute path to the
        # pipeline_*.py files.  This dir is also on sys.path via src/__init__.py.
        custom_pipeline = str(
            cfg.third_party_dir / "Hunyuan3D-2.1" / "hy3dpaint" / "hunyuanpaintpbr"
        )

        logger.info("Loading HunyuanPaintPipeline from %s ...", model_dir)
        pipeline = DiffusionPipeline.from_pretrained(
            str(model_dir),
            custom_pipeline=custom_pipeline,
            torch_dtype=torch.float16,
        )

        # UniPC scheduler (upstream default, 15 steps at good quality)
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing",
        )

        pipeline.enable_vae_slicing()
        pipeline.enable_vae_tiling()
        pipeline.eval()

        # Attach view_size from YAML so forward can reference it
        setattr(pipeline, "view_size", cfg_view_size)
        self.pipeline = pipeline.to(cfg.device)

        # ---- conditional DINOv2 loading --------------------------------------
        self.dino = None
        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2  # type: ignore[import]

            dino_dir = cfg.models_dir / "dinov2-giant"
            if not dino_dir.exists():
                raise FileNotFoundError(
                    f"DINOv2 model not found: {dino_dir}\n"
                    "The paint UNet requires DINOv2.  "
                    "Run:  python scripts/download_models.py"
                )
            logger.info("Loading DINOv2 from %s ...", dino_dir)
            self.dino = Dino_v2(str(dino_dir))
            self.dino = self.dino.to(device=cfg.device, dtype=torch.float16)

        logger.info(
            "MultiviewDiffusionNet ready.  mode=%s  view_size=%d  scheduler=%s.",
            self.mode, cfg_view_size,
            type(self.pipeline.scheduler).__name__,
        )

    def __call__(
        self,
        reference_rgb: Image.Image,
        normal_maps: List[Image.Image],
        position_maps: List[Image.Image],
        cfg,
    ) -> Dict[str, List[Image.Image]]:
        """
        Run multiview PBR diffusion to generate albedo and MR texture views.

        Condition images are structured as:
            [normal_0, ..., normal_N, position_0, ..., position_N]
        and split at len//2 before being passed to the pipeline — matching
        the structure produced by render_multiview.render_conditioning_maps.

        The front-view reference is the appearance anchor.  It is passed as
        input_images[0:1] (first element only) — the pipeline uses it as the
        DINO feature source and appearance conditioning signal.

        Args:
            reference_rgb:  Gray-composited front-view reference (RGB).
            normal_maps:    Normal-map PIL images from render_conditioning_maps.
            position_maps:  Position-map PIL images from render_conditioning_maps.
            cfg:            PipelineConfig with paint_steps, paint_guidance_scale, seed.

        Returns:
            Dict with:
              "albedo": List[PIL.Image] — one per camera view
              "mr":     List[PIL.Image] — metallic-roughness, one per camera view
        """
        import torch

        view_size: int = self.pipeline.view_size

        # Resize reference to model input resolution
        input_img = reference_rgb.resize((view_size, view_size)).convert("RGB")

        # Concatenate normal + position maps and resize to view_size
        control_images: List[Image.Image] = []
        for img in normal_maps + position_maps:
            resized = img.resize((view_size, view_size))
            if resized.mode == "L":
                resized = resized.point(lambda x: 255 if x > 1 else 0, mode="1")
            control_images.append(resized)

        num_view = len(control_images) // 2
        normal_batch = [[control_images[i] for i in range(num_view)]]
        position_batch = [[control_images[i + num_view] for i in range(num_view)]]

        seed = int(cfg.seed) % (2 ** 32)
        kwargs: dict = {
            "width": view_size,
            "height": view_size,
            "num_in_batch": num_view,
            "images_normal": normal_batch,
            "images_position": position_batch,
            "generator": torch.Generator(device=self.device).manual_seed(seed),
        }

        if self.dino is not None:
            kwargs["dino_hidden_states"] = self.dino(input_img)

        logger.info(
            "Running multiview diffusion: %d views  steps=%d  guidance=%.1f  seed=%d.",
            num_view, cfg.paint_steps, cfg.paint_guidance_scale, seed,
        )

        with torch.inference_mode():
            outputs = self.pipeline(
                [input_img],
                num_inference_steps=cfg.paint_steps,
                prompt="high quality",
                sync_condition=None,
                guidance_scale=cfg.paint_guidance_scale,
                **kwargs,
            ).images

        if "pbr" in self.mode:
            result: Dict[str, List[Image.Image]] = {
                "albedo": outputs[:num_view],
                "mr": outputs[num_view:],
            }
        else:
            result = {"albedo": outputs, "mr": outputs}

        logger.info(
            "Diffusion complete: %d albedo views, %d MR views.",
            len(result["albedo"]), len(result["mr"]),
        )
        return result

    def unload(self) -> None:
        """Delete pipeline + DINOv2 and release GPU memory."""
        del self.pipeline
        if self.dino is not None:
            del self.dino
        self.dino = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("MultiviewDiffusionNet unloaded from GPU.")


# ---------------------------------------------------------------------------
# Step L: Upscale generated views
# ---------------------------------------------------------------------------

def upscale_views(
    images: List[Image.Image],
    target_size: int,
    *,
    use_realesrgan: bool = False,
    realesrgan_ckpt: "Path | str | None" = None,
) -> List[Image.Image]:
    """
    Upscale multiview texture views to target_size before UV baking.

    Higher resolution at bake time reduces aliasing in the UV texture atlas.
    target_size should be cfg.render_size (typically 2× texture_size).

    Two modes:
      Lanczos (default): CPU, deterministic, fast.
      RealESRGAN 4×:     GPU, higher quality — requires RealESRGAN_x4plus.pth.
                         After 4× upscale the result is resized with Lanczos
                         to exactly target_size (in case the image was not a
                         clean multiple of 4).

    Args:
        images:          List of PIL images to upscale.
        target_size:     Output width and height in pixels (cfg.render_size).
        use_realesrgan:  Use RealESRGAN instead of Lanczos.
        realesrgan_ckpt: Path to RealESRGAN_x4plus.pth checkpoint.
                         Required when use_realesrgan=True.

    Returns:
        List of PIL.Image resized to (target_size, target_size).
    """
    if not images:
        return images

    if use_realesrgan:
        if realesrgan_ckpt is None:
            raise ValueError(
                "use_realesrgan=True requires realesrgan_ckpt path to be provided."
            )
        from utils.image_super_utils import imageSuperNet  # type: ignore[import]

        super_cfg = SimpleNamespace(realesrgan_ckpt_path=str(realesrgan_ckpt))
        super_model = imageSuperNet(super_cfg)

        logger.info(
            "Upscaling %d views with RealESRGAN → %d px.", len(images), target_size
        )
        upscaled = [super_model(img) for img in images]
        # Resize to exact target in case 4× overshot or image wasn't square
        return [
            img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            for img in upscaled
        ]

    logger.info(
        "Upscaling %d views with Lanczos → %d px.", len(images), target_size
    )
    return [
        img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        for img in images
    ]
