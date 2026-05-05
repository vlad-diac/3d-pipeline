"""
Pipeline configuration dataclasses.

CameraConfig: virtual camera layout for multiview texture rendering (6 views).
PipelineConfig: all runtime parameters — paths, device, shape/paint/inpaint settings.

Path derivation is automatic from the file's location in the src/ package.
Override project_root if running from a non-standard working directory.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class CanonicalMultiviewConfig:
    """
    Configuration for the optional canonical multiview generation stage.

    Uses MV-Adapter i2mv SDXL with an optional depth / canny ControlNet branch
    to synthesise consistent front / right / back / left views from a single
    clean RGBA anchor image produced by the ``remove_background`` stage.
    """

    enabled: bool = False

    # ------------------------------------------------- model identifiers
    adapter_repo: str = "huanngzh/mv-adapter"
    adapter_weight: str = "mvadapter_i2mv_sdxl_beta.safetensors"
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix"

    # ------------------------------------------- generation parameters
    gen_size: int = 768
    steps: int = 50
    guidance_scale: float = 3.0
    reference_conditioning_scale: float = 1.0

    # -------------------------------------------- structural controls
    use_depth: bool = True
    depth_scale: float = 0.5
    use_canny: bool = False
    canny_scale: float = 0.2

    # ------------------------------------------------ diffusion prompt
    prompt: str = "high quality"
    negative_prompt: str = "watermark, ugly, deformed, noisy, blurry, low contrast"


@dataclass
class CameraConfig:
    """
    Six-camera layout for multiview texture rendering and baking.

    View assignments:
        0 — Front  (az=0°,   el=0°)    weight=1.00  dominant appearance anchor
        1 — Right  (az=90°,  el=0°)    weight=0.10  side coverage
        2 — Back   (az=180°, el=0°)    weight=0.50  back coverage
        3 — Left   (az=270°, el=0°)    weight=0.10  side coverage
        4 — Top    (az=0°,   el=90°)   weight=0.05  polar cap
        5 — Bottom (az=180°, el=-90°)  weight=0.05  polar cap

    Weights control cosine-weighted blending priority during UV back-projection.
    The front view has the highest trust because it matches the input image most closely.

    IMPORTANT: View order must stay consistent across render → paint → bake stages.
    """

    azimuths: List[int] = field(default_factory=lambda: [0, 90, 180, 270, 0, 180])
    elevations: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 90, -90])
    weights: List[float] = field(default_factory=lambda: [1.0, 0.1, 0.5, 0.1, 0.05, 0.05])

    # Orthographic camera frustum scale.
    # MeshRender requires this explicitly — omitting it causes wrong projection scale.
    ortho_scale: float = 1.0

    def __post_init__(self) -> None:
        n = len(self.azimuths)
        if len(self.elevations) != n or len(self.weights) != n:
            raise ValueError(
                f"azimuths ({n}), elevations ({len(self.elevations)}), and "
                f"weights ({len(self.weights)}) must all have the same length."
            )
        if n < 1:
            raise ValueError("CameraConfig requires at least one view.")

    @classmethod
    def four_cardinal(cls) -> "CameraConfig":
        """
        Four-camera layout matching MV-Adapter's canonical azimuths.

        Used when the ``canonical_multiview`` stage is active.  The four
        cardinal views (front/right/back/left at 0/90/180/270°) replace the
        default six-camera layout.  Top and bottom texels that would have been
        covered by the top/bottom cameras are filled by the inpaint stage.
        """
        return cls(
            azimuths=[0, 90, 180, 270],
            elevations=[0, 0, 0, 0],
            weights=[1.0, 0.4, 0.6, 0.4],
        )


@dataclass
class PipelineConfig:
    """Full runtime configuration for the Hunyuan3D multiview pipeline."""

    # ------------------------------------------------------------------ paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    # Derived paths — set in __post_init__ so they track project_root changes.
    models_dir: Path = field(init=False)
    third_party_dir: Path = field(init=False)
    delight_model_dir: Path = field(init=False)

    # ----------------------------------------------------------------- device
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # -------------------------------------------------------- shape generation
    # Number of diffusion steps for the shape DiT.
    shape_steps: int = 50
    shape_guidance_scale: float = 5.0
    # Octree resolution for the marching-cubes mesh extraction.
    # 384 is the default for v2.1; higher = more detail but more VRAM.
    octree_resolution: int = 384

    # When True, use the v2.0 four-view model (Path B).
    # When False, use the v2.1 single-image model (Path A).
    use_multiview_shape: bool = False
    # HuggingFace subfolder for the four-view shape model.
    shape_model_subfolder: str = "hunyuan3d-dit-v2-mv"

    # ------------------------------------------------------ mesh postprocessing
    target_faces: int = 200_000
    normalize_mesh: bool = True

    # ---------------------------------------------------------------- cameras
    camera: CameraConfig = field(default_factory=CameraConfig)

    # ------------------------------------------ canonical multiview stage
    canonical: CanonicalMultiviewConfig = field(
        default_factory=CanonicalMultiviewConfig
    )

    # ----------------------------------------------- paint / texture settings
    # Diffusion view resolution (model input size).
    view_size: int = 512
    # Resolution of rendered conditioning maps (normals, positions).
    # Must be >= texture_size for baking quality.
    render_size: int = 2048
    # Final UV texture atlas resolution.
    texture_size: int = 4096

    paint_steps: int = 10
    paint_guidance_scale: float = 3.0

    # Enable optional delight (Light_Shadow_Remover) step.
    use_delight: bool = True
    # Use RealESRGAN for view upscaling instead of Lanczos.
    use_realesrgan: bool = False

    # --------------------------------------------------------------- inpainting
    # Enable vertex-aware inpainting (pass 1) before cv2 inpainting (pass 2).
    vertex_inpaint: bool = True
    # "NS" = Navier-Stokes, "TELEA" = Fast Marching
    inpaint_method: str = "NS"

    # ---------------------------------------------------- background removal
    # Pixels to erode from the rembg alpha mask edge after background removal.
    # Clips halo bleed where rembg leaves a semi-transparent fringe around the
    # foreground silhouette.  0 = disabled.  Start with 1–2 for real photographs;
    # leave at 0 for clean CG renders where rembg already gives sharp edges.
    rembg_erode_px: int = 0

    # ---------------------------------------------------- subject centering
    # When True, crop to subject bounding box, extend to square, add margin.
    # Ensures consistent framing for the shape model's DINOv2 image encoder.
    preprocess_center_subject: bool = True

    # ------------------------------------------------ canonical resize
    # Resize the RGBA subject to this square side length after centering.
    # 518 px matches the DINOv2-Giant encoder expectation documented in the
    # ComfyUI pipeline ("tied to the image encoder").
    # None = no resize (keep size produced by center_and_pad).
    preprocess_target_size: int | None = None

    # ---------------------------------------------------------- reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        self.models_dir = self.project_root / "models"
        self.third_party_dir = self.project_root / "third_party"
        self.delight_model_dir = self.models_dir / "delight"

        if self.render_size < self.texture_size:
            import warnings
            warnings.warn(
                f"render_size ({self.render_size}) is smaller than texture_size "
                f"({self.texture_size}). Baked textures will be aliased. "
                "Set render_size >= texture_size.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------helpers

    @classmethod
    def for_macos_dev(cls) -> "PipelineConfig":
        """CPU-safe configuration for local macOS development and import testing."""
        return cls(
            device="cpu",
            dtype=torch.float32,
            shape_steps=1,
            paint_steps=1,
            octree_resolution=128,
            target_faces=10_000,
            view_size=256,
            render_size=512,
            texture_size=512,
            use_delight=False,
            use_realesrgan=False,
            vertex_inpaint=False,
        )

    @classmethod
    def for_runpod(cls, texture_size: int = 4096) -> "PipelineConfig":
        """Full-quality GPU configuration for RunPod (CUDA 12.4, >=24 GB VRAM)."""
        return cls(
            device="cuda",
            dtype=torch.float16,
            texture_size=texture_size,
            render_size=max(2048, texture_size),
        )
