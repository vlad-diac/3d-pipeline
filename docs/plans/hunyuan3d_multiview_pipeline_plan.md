# Consolidated Hunyuan3D Multiview Pipeline — Final Plan

> Synthesized from three source documents and two reviews. Every decision is justified by cross-referencing the visualbruno ComfyUI wrapper, the official Tencent codebase, and verified HuggingFace model trees.

---

## Architecture Decision: Two Supported Geometry Paths

| Path | Shape Model | Input | Status |
|------|-------------|-------|--------|
| **A. Single-Image** | `hunyuan3d-dit-v2-1` (3.3B, v2.1) | One RGBA image | Production-ready |
| **B. Four-View** | `hunyuan3d-dit-v2-mv` (v2.0 2mv) | front + left + right + back | Production-ready via `from_pretrained` |

Both paths converge into the **same texture pipeline** (Hunyuan3D-Paint-PBR v2.1) after mesh generation.

---

## 1. Repositories Required

| Repository | Purpose | What We Use |
|------------|---------|-------------|
| [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Official 2.1 shape + paint pipeline source | `hy3dshape/`, `hy3dpaint/`, CUDA extensions |
| [Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | Official 2.0 codebase — used for multiview shape + Delight utility | `hy3dgen/texgen/utils/dehighlight_utils.py` |
| [visualbruno/ComfyUI-Hunyuan3d-2-1](https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1) | Reference for split-flow architecture | `nodes.py` (architecture reference only), `configs/dit_config_2_1.yaml` |

**Clone layout:**
```
third_party/
├── Hunyuan3D-2.1/        # Primary runtime dependency
├── Hunyuan3D-2/          # For Delight model + 2mv shape if needed
└── ComfyUI-Hunyuan3d-2-1/  # Architecture reference only
```

---

## 2. Model Weights Required

| Model | HuggingFace Location | Size | Purpose |
|-------|---------------------|------|---------|
| Hunyuan3D-Shape v2.1 | `tencent/Hunyuan3D-2.1` / `hunyuan3d-dit-v2-1/` | ~7.4 GB | Single-image shape DiT |
| Hunyuan3D-VAE v2.1 | `tencent/Hunyuan3D-2.1` / `hunyuan3d-vae-v2-1/` | ~656 MB | Latent → mesh decoder (separate checkpoint) |
| Hunyuan3D-Paint-PBR v2.1 | `tencent/Hunyuan3D-2.1` / `hunyuan3d-paintpbr-v2-1/` | ~4 GB | Multiview texture diffusion |
| DINOv2-Giant | `facebook/dinov2-giant` | ~4.4 GB | Image feature conditioning for paint UNet |
| Hunyuan3D-2mv Shape | `tencent/Hunyuan3D-2mv` / `hunyuan3d-dit-v2-mv/` | ~7 GB | Four-view shape DiT (optional Path B) |
| Delight Model | TBD (from Hunyuan3D-2 release) | ~2 GB | Light/shadow removal (optional quality step) |
| RealESRGAN x4plus | Direct download | ~64 MB | Texture view upscaling |

**Storage layout:**
```
models/
├── hunyuan3d-dit-v2-1/        # config.yaml + model.fp16.ckpt
├── hunyuan3d-vae-v2-1/        # model.fp16.ckpt
├── hunyuan3d-paintpbr-v2-1/   # Full diffusers pipeline folder
├── hunyuan3d-dit-v2-mv/       # (Path B) multiview shape model
├── dinov2-giant/              # DINOv2 weights
├── delight/                   # Light_Shadow_Remover checkpoint (optional)
└── RealESRGAN_x4plus.pth     # Upscaler
```

---

## 3. Libraries and Versions

### Core Stack (Tested and Verified)

| Package | Version | Source |
|---------|---------|--------|
| Python | 3.10 | Official tested stack |
| PyTorch | 2.5.1+cu124 | Official tested stack |
| torchvision | 0.20.1+cu124 | Matches PyTorch |
| torchaudio | 2.5.1+cu124 | Matches PyTorch |
| CUDA Toolkit | 12.4 | Required for extension builds |

### ML / Diffusion

| Package | Version | Why |
|---------|---------|-----|
| diffusers | 0.31.0 | Paint pipeline base (Research 1 pin) |
| transformers | 4.46.0 | DINO + conditioner |
| accelerate | 1.1.1 | Model offloading |
| safetensors | 0.4.5 | Checkpoint loading |
| huggingface-hub | 0.30.2 | Model download |
| pytorch-lightning | 2.4.0 | Shape pipeline dependency |
| timm | 1.0.11 | Vision backbone utilities |
| einops | 0.8.0 | Tensor reshaping in models |

### Image Processing

| Package | Version | Why |
|---------|---------|-----|
| opencv-python | 4.10.0.84 | Inpainting (cv2.INPAINT_NS) |
| Pillow | 10.4.0 | Image I/O |
| scikit-image | 0.24.0 | Image processing utilities |
| rembg | 2.0.65 | Background removal |
| onnxruntime | 1.19.2 | Required by rembg |
| basicsr | 1.4.2 | Required by realesrgan |
| realesrgan | 0.3.0 | Texture view upscaling |

### 3D Mesh Processing

| Package | Version | Why |
|---------|---------|-----|
| trimesh | 4.4.9 | Mesh representation + export |
| xatlas | **0.0.10** | UV unwrapping (official 2.1 pin) |
| pymeshlab | 2023.12.post2 | Mesh decimation fallback |
| pygltflib | 1.16.3 | GLB assembly with PBR materials |
| open3d | 0.18.0 | Mesh utilities |

### Configuration / Build

| Package | Version | Why |
|---------|---------|-----|
| omegaconf | 2.3.0 | Paint model config loading |
| pyyaml | 6.0.2 | YAML config parsing |
| configargparse | 1.7 | Argument parsing |
| pybind11 | 2.13.6 | C++ extension builds |
| ninja | 1.11.1.1 | Fast extension compilation |
| cupy-cuda12x | 13.3.0 | GPU mesh operations |

### Optional (Quality Enhancement)

| Package | Version | Why |
|---------|---------|-----|
| xformers | 0.0.28.post3 | Optional attention speedup (not required; SDPA is default) |

---

## 4. CUDA Extensions (Must Be Compiled)

| Extension | Location | Build Command | Purpose |
|-----------|----------|---------------|---------|
| `custom_rasterizer_kernel` | `hy3dpaint/custom_rasterizer/` | `pip install -e .` | GPU mesh rasterization for normal/position maps and baking |
| `mesh_inpaint_processor` | `hy3dpaint/DifferentiableRenderer/` | `bash compile_mesh_painter.sh` | Vertex-aware texture inpainting |

**Build prerequisites:**
- `nvcc` available (from CUDA Toolkit 12.4)
- `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` set for your GPU
- `ninja-build` installed at system level
- `pybind11` installed in environment

---

## 5. Project Structure

```
hunyuan3d-multiview/
├── setup.sh                      # Full environment bootstrap
├── requirements.txt              # Pinned Python dependencies
├── main.py                       # Root entry point
├── src/
│   ├── __init__.py
│   ├── config.py                 # All paths, params, device config
│   ├── preprocess.py             # Image loading, background removal, compose_over_gray
│   ├── mesh_generate.py          # Shape pipeline (single-image or 4-view)
│   ├── mesh_postprocess.py       # Floater removal, decimation, normalization
│   ├── render_multiview.py       # UV unwrap, normal/position map rendering
│   ├── paint_multiview.py        # Delight, multiview diffusion, upscaling
│   ├── bake_texture.py           # Cosine-weighted UV back-projection
│   ├── inpaint_texture.py        # Vertex-aware + cv2 inpainting
│   └── export_glb.py             # PBR material assembly + GLB export
├── scripts/
│   └── download_models.py        # HuggingFace model downloader
├── inputs/                       # Input images
├── outputs/                      # Final outputs + numbered debug dirs
├── models/                       # Downloaded weights
└── third_party/                  # Cloned repositories
    ├── Hunyuan3D-2.1/
    └── Hunyuan3D-2/
```

---

## 6. Step-by-Step Implementation Plan

### Overview Flow Diagram

```
Input Image(s)
    │
    ▼
[A] Load & Validate Images
    │
    ▼
[B] Preprocess (rembg → RGBA → compose over gray)
    │
    ▼
[C] Load Shape Pipeline (v2.1 single-image OR v2.0 2mv)
    │
    ▼
[D] Generate Mesh (from_pretrained → trimesh output)
    │       └─ [D-alt] Latent split: generate latent → load VAE → decode
    ▼
[E] Postprocess Mesh (floaters, degenerate, decimate, normalize)
    │
    ▼
[F] UV Unwrap (xatlas)
    │
    ▼
[G] Build Camera Config (6 views: 4 cardinal + top + bottom)
    │
    ▼
[H] Init Paint Pipeline (MeshRender + ViewProcessor + MultiviewDiffusionNet)
    │
    ▼
[I] Render Normal Maps + Position Maps (6 views from UV mesh)
    │
    ▼
[J] Delight Reference Image (optional Light_Shadow_Remover)
    │
    ▼
[K] Run Multiview Paint Diffusion (→ 6× albedo + 6× metallic-roughness)
    │
    ▼
[L] Upscale Views (RealESRGAN or Lanczos to render_size)
    │
    ▼
[M] Bake into UV Textures (cosine-weighted back-projection)
    │
    ▼
[N] Inpaint Missing Texels (vertex-aware → cv2.INPAINT_NS)
    │
    ▼
[O] Apply Textures + Export GLB (create_glb_with_pbr_materials)
```

---

### Step A: Load and Validate Input Images

**Purpose:** Load one or more input images, validate they are usable.

**Implementation:**
```python
# src/preprocess.py

from pathlib import Path
from PIL import Image

def load_image_rgba(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGBA")
    if img.getbbox() is None:
        raise ValueError(f"Image is fully transparent: {path}")
    return img

def collect_views(args) -> dict:
    """Build the view dictionary for geometry generation."""
    views = {}
    for key in ("front", "left", "right", "back"):
        path = getattr(args, key, None)
        if path:
            views[key] = load_image_rgba(Path(path))
    
    if not views and args.image:
        views["front"] = load_image_rgba(Path(args.image))
    
    if not views:
        raise ValueError("Provide --image or --front at minimum.")
    
    return views
```

**Decisions:**
- Always convert to RGBA on load (Research 1 pattern)
- Validate non-empty image (`getbbox()` check from Research 1)
- Support both single-image and four-view inputs
- No EXIF handling needed for synthetic renders; add `ImageOps.exif_transpose()` if processing photographs

---

### Step B: Preprocess — Background Removal + Gray Composite

**Purpose:** Isolate the subject and create a neutral appearance reference for the paint model.

**Implementation:**
```python
# src/preprocess.py

import numpy as np

def maybe_remove_background(
    image: Image.Image, 
    remove_bg: bool, 
    background_remover=None
) -> Image.Image:
    """Only run rembg if alpha channel is fully opaque (no existing transparency)."""
    image = image.convert("RGBA")
    alpha_min, alpha_max = image.getchannel("A").getextrema()
    
    if alpha_min == alpha_max == 255 and remove_bg:
        if background_remover is None:
            from hy3dshape.rembg import BackgroundRemover
            background_remover = BackgroundRemover()
        image = background_remover(image)
    
    return image

def compose_over_gray(image: Image.Image, gray: int = 127) -> Image.Image:
    """Composite RGBA onto neutral gray background.
    
    Gray (not white) reduces directional lighting bias in the paint model,
    giving more balanced texture generation.
    """
    image = image.convert("RGBA")
    bg = Image.new("RGBA", image.size, (gray, gray, gray, 255))
    composed = Image.alpha_composite(bg, image)
    return composed.convert("RGB")

def compose_over_white(image: Image.Image) -> Image.Image:
    """Composite RGBA onto white background (alternative for shape conditioning)."""
    image = image.convert("RGBA")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(bg, image)
    return composed.convert("RGB")
```

**Decisions:**
- **Gray composite for paint reference** (Research 2 finding: reduces baked-in lighting)
- **Alpha-extrema check** before running rembg (Research 1 pattern: don't destroy existing masks)
- **White composite for shape conditioning** (shape model was trained with white-bg images)
- Lazy-load `BackgroundRemover` to avoid import overhead when not needed

---

### Step C: Load Shape Pipeline

**Purpose:** Initialize the geometry generation model.

**Implementation:**
```python
# src/mesh_generate.py

import torch
from pathlib import Path

def load_shape_pipeline(cfg):
    """
    Load the shape pipeline using from_pretrained (stable public API).
    
    Supports two modes:
    - Single-image: tencent/Hunyuan3D-2.1, subfolder hunyuan3d-dit-v2-1
    - Four-view:    tencent/Hunyuan3D-2mv, subfolder hunyuan3d-dit-v2-mv
    """
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    
    if cfg.use_multiview_shape:
        # Path B: four-view geometry (v2.0 2mv model)
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2mv',
            subfolder=cfg.shape_model_subfolder,  # 'hunyuan3d-dit-v2-mv'
            device=cfg.device,
            dtype=torch.float16,
        )
    else:
        # Path A: single-image geometry (v2.1)
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            subfolder='hunyuan3d-dit-v2-1',
            device=cfg.device,
            dtype=torch.float16,
        )
    
    return pipeline
```

**Decisions:**
- **Use `from_pretrained`** — Research 2 identifies this as the stable public API surface
- Support both single-image and 2mv paths in one function
- Pass `dtype=torch.float16` explicitly (Research 1 finding: Tutorial omits this)
- The `from_pretrained` path handles VAE loading internally, avoiding the need for separate VAE management

**Fallback for local checkpoints** (when offline or on custom weights):
```python
def load_shape_pipeline_local(cfg):
    """For offline / custom checkpoint usage."""
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    
    ckpt_path = cfg.models_dir / "hunyuan3d-dit-v2-1" / "model.fp16.ckpt"
    config_path = cfg.models_dir / "hunyuan3d-dit-v2-1" / "config.yaml"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing shape checkpoint: {ckpt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing shape config: {config_path}")
    
    return Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=str(ckpt_path),
        config_path=str(config_path),
        device=cfg.device,
        dtype=torch.float16,
        use_safetensors=False,
    )
```

---

### Step D: Generate Mesh

**Purpose:** Run shape diffusion and produce a triangle mesh.

**Implementation:**
```python
# src/mesh_generate.py

import gc
import trimesh

def generate_mesh(pipeline, views: dict, cfg) -> trimesh.Trimesh:
    """
    Generate a 3D mesh from input view(s).
    
    For single-image: pass the front view PIL image.
    For four-view: pass a dict with keys {front, left, right, back}.
    
    Uses the public API (output_type="trimesh") which returns a mesh directly.
    This avoids the internal/unstable latent split flow.
    """
    seed = cfg.seed % (2**32)
    generator = torch.Generator(device=cfg.device).manual_seed(seed)
    
    # Determine input format
    if len(views) == 1:
        image_input = list(views.values())[0]
    elif len(views) >= 2:
        # Four-view dict input for 2mv model
        image_input = views
    else:
        raise ValueError("No views provided")
    
    with torch.inference_mode():
        result = pipeline(
            image=image_input,
            num_inference_steps=cfg.shape_steps,
            guidance_scale=cfg.shape_guidance_scale,
            generator=generator,
            octree_resolution=cfg.octree_resolution,
            output_type="trimesh",
        )
    
    mesh = result[0] if isinstance(result, list) else result
    
    # Cleanup shape pipeline memory
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    return mesh
```

**Decisions:**
- **Use `output_type="trimesh"`** — the public API returns a mesh directly, avoiding unstable latent APIs (Research 2 recommendation)
- Support both single-image (PIL) and multi-view (dict) inputs transparently
- Use `torch.inference_mode()` (Research 2: more efficient than `@torch.no_grad()`)
- Clean up GPU memory immediately after shape generation (shape uses ~10 GB)
- Use `torch.Generator(device=...)` for proper device-aware seeding (Research 1 finding)

**Alternative: Explicit latent split (advanced/custom usage only):**
```python
def generate_mesh_via_latent_split(pipeline, vae, image, cfg):
    """
    ADVANCED: Separate latent generation from VAE decode.
    Only use this if you need to cache/modify latents between steps.
    WARNING: output_type="latent" is not guaranteed stable across releases.
    """
    with torch.inference_mode():
        try:
            latents = pipeline(
                image=image,
                num_inference_steps=cfg.shape_steps,
                guidance_scale=cfg.shape_guidance_scale,
                generator=torch.Generator(device=cfg.device).manual_seed(cfg.seed),
                output_type="latent",
            )
        except TypeError:
            # Fallback: pipeline doesn't support output_type kwarg
            raise RuntimeError("This pipeline version does not support latent output")
    
    # Decode with separate VAE
    latents = vae.decode(latents)
    outputs = vae.latents2mesh(
        latents,
        output_type='trimesh',
        bounds=1.01,
        mc_level=0.0,
        num_chunks=8000,
        octree_resolution=cfg.octree_resolution,
        mc_algo='mc',
        enable_pbar=True,
    )[0]
    
    outputs.mesh_f = outputs.mesh_f[:, ::-1]
    return trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f, process=False)
```

---

### Step E: Postprocess Mesh

**Purpose:** Clean geometry artifacts and normalize for consistent renderer behavior.

**Implementation:**
```python
# src/mesh_postprocess.py

import numpy as np
import trimesh

def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if not parts:
        return mesh
    return max(parts, key=lambda m: len(m.faces))

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center at origin, scale to unit bounding box."""
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    scale = float(np.max(bounds[1] - bounds[0]))
    if scale <= 1e-8:
        return mesh
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale)
    return mesh

def postprocess_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int = 200_000,
    normalize: bool = True,
) -> trimesh.Trimesh:
    """
    Full mesh cleanup pipeline:
    1. Remove infinite/NaN vertices
    2. Remove unreferenced vertices and duplicate/degenerate faces
    3. Merge coincident vertices
    4. Keep largest connected component (remove floaters)
    5. Optionally decimate to target face count
    6. Normalize to unit bounding box
    """
    mesh = mesh.copy()
    
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.merge_vertices()
    
    mesh = keep_largest_component(mesh)
    
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
        except Exception:
            # Fallback: pymeshlab decimation
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
                face_matrix=np.asarray(mesh.faces, dtype=np.int32),
            ))
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target_faces,
                preservetopology=True,
                preserveboundary=True,
            )
            current = ms.current_mesh()
            mesh = trimesh.Trimesh(
                vertices=current.vertex_matrix(),
                faces=current.face_matrix(),
                process=False,
            )
    
    if normalize:
        mesh = normalize_mesh(mesh)
    
    mesh.remove_unreferenced_vertices()
    return mesh
```

**Decisions:**
- **Include `normalize_mesh`** (Research 2: ensures consistent renderer scale)
- **200k face target** (Research 1 + 2 agree; Tutorial's 40k is too aggressive for texture quality)
- **Fallback decimation chain**: trimesh built-in → pymeshlab (handles edge cases)
- `process=False` on trimesh construction (Research 1: prevents unwanted geometry modification)
- `remove_infinite_values()` (Research 2: catches marching-cubes artifacts)

---

### Step F: UV Unwrap

**Purpose:** Generate UV coordinates for texture mapping via xatlas.

**Implementation:**
```python
# src/render_multiview.py

import xatlas
import trimesh

def uv_unwrap_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Apply xatlas UV parametrization.
    
    MUST happen before:
    - Normal/position map rendering (renderer needs UVs to know which texel corresponds to which surface point)
    - Texture baking (projects view colors into UV space)
    - Inpainting (operates in UV texture space)
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    if len(mesh.faces) > 500_000_000:
        raise ValueError("Mesh exceeds 500M faces — too large for xatlas")
    
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs
    
    return mesh
```

**Decisions:**
- Uses `xatlas==0.0.10` (Research 2: official 2.1 pin)
- Same implementation across all three source documents — this step is well-agreed
- Handle `trimesh.Scene` case (upstream `mesh_uv_wrap` does this)

---

### Step G: Camera Configuration

**Purpose:** Define the 6 virtual cameras used for multiview texture rendering and baking.

**Implementation:**
```python
# src/config.py (partial)

from dataclasses import dataclass, field
from typing import List

@dataclass
class CameraConfig:
    """
    Default 6-camera setup:
        View 0: Front  (azim=0°,   elev=0°)   weight=1.0   — dominant view
        View 1: Right  (azim=90°,  elev=0°)   weight=0.1   — side coverage
        View 2: Back   (azim=180°, elev=0°)   weight=0.5   — back coverage
        View 3: Left   (azim=270°, elev=0°)   weight=0.1   — side coverage
        View 4: Top    (azim=0°,   elev=90°)  weight=0.05  — polar cap
        View 5: Bottom (azim=180°, elev=-90°) weight=0.05  — polar cap
    
    Weights control blending priority during UV back-projection.
    Front view gets highest trust (matches input image most closely).
    """
    azimuths: List[int] = field(default_factory=lambda: [0, 90, 180, 270, 0, 180])
    elevations: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 90, -90])
    weights: List[float] = field(default_factory=lambda: [1.0, 0.1, 0.5, 0.1, 0.05, 0.05])
    ortho_scale: float = 1.0
```

**Why 6 cameras when geometry uses 4 views:**
The geometry stage uses 4 external photographs (front/left/right/back) to build the 3D shape. The texture stage then renders 6 **internal geometry views** (4 cardinal + top + bottom) from the UV-unwrapped mesh. These are geometric conditioning signals (normals + positions), not appearance references. The appearance anchor is always the single front view.

---

### Step H: Initialize Paint Pipeline

**Purpose:** Set up the renderer, view processor, and multiview diffusion model.

**Implementation:**
```python
# src/render_multiview.py

import torch
from pathlib import Path
from DifferentiableRenderer.MeshRender import MeshRender
from utils.pipeline_utils import ViewProcessor

class PaintPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.camera = cfg.camera
        
        # Initialize renderer WITH ortho_scale
        self.render = MeshRender(
            default_resolution=cfg.render_size,
            texture_size=cfg.texture_size,
            bake_mode="back_sample",
            raster_mode="cr",
            ortho_scale=cfg.camera.ortho_scale,  # CRITICAL: must pass this
        )
        
        # View processor handles rendering + baking operations
        # Build a config object the ViewProcessor expects
        class _PaintConf:
            pass
        
        paint_conf = _PaintConf()
        paint_conf.device = cfg.device
        paint_conf.render_size = cfg.render_size
        paint_conf.texture_size = cfg.texture_size
        paint_conf.bake_exp = 4
        paint_conf.merge_method = "fast"
        paint_conf.raster_mode = "cr"
        paint_conf.bake_mode = "back_sample"
        paint_conf.resolution = cfg.view_size
        paint_conf.ortho_scale = cfg.camera.ortho_scale
        
        self.view_processor = ViewProcessor(paint_conf, self.render)
        self.multiview_model = None  # Lazy-loaded in Step K
    
    def load_mesh(self, mesh):
        self.render.load_mesh(mesh=mesh)
```

**Decisions:**
- **Pass `ortho_scale`** to `MeshRender` constructor (Research 1 + 2: Tutorial omits this — critical bug)
- **Set `bake_exp=4`** (Research 1: controls cosine-weight exponent during baking)
- **Set `merge_method="fast"`** (upstream default for texture blending)
- Lazy-load multiview model to separate memory pressure between stages

---

### Step I: Render Normal and Position Maps

**Purpose:** Rasterize the UV-unwrapped mesh from each camera to produce conditioning images.

**Implementation:**
```python
# src/render_multiview.py (continued)

@torch.inference_mode()
def render_conditioning_maps(pipeline: PaintPipeline, cfg):
    """
    Render normal maps and position maps from each camera view.
    
    These condition the paint diffusion model:
    - Normals: surface orientation (RGB = Nx, Ny, Nz) — tells model where edges/creases are
    - Positions: world-space coordinates (normalized RGB) — ensures spatial consistency across views
    
    The condition tensor is structured as:
        [normal_0, normal_1, ..., normal_5, position_0, position_1, ..., position_5]
    This ordering is CRITICAL — the multiview model splits at len//2.
    """
    normal_maps = pipeline.view_processor.render_normal_multiview(
        cfg.camera.elevations,
        cfg.camera.azimuths,
        use_abs_coor=True,
    )
    position_maps = pipeline.view_processor.render_position_multiview(
        cfg.camera.elevations,
        cfg.camera.azimuths,
    )
    return normal_maps, position_maps
```

**Key constraint:** The view order must remain consistent through rendering → paint generation → baking. If order changes between stages, textures will be misprjected.

---

### Step J: Delight Reference Image (Optional Quality Step)

**Purpose:** Remove directional lighting from the reference image before paint diffusion, preventing baked-in highlights/shadows.

**Implementation:**
```python
# src/paint_multiview.py

from types import SimpleNamespace

def delight_reference(front_rgba: Image.Image, cfg) -> Image.Image:
    """
    Remove lighting/shadow bias from the reference image.
    
    Process:
    1. Composite front view onto neutral gray (not white) background
    2. Run Light_Shadow_Remover to produce a flat-lit appearance reference
    
    If the delight model is unavailable, returns the gray composite directly.
    This is optional but improves quality significantly on real-world photos.
    """
    reference = compose_over_gray(front_rgba, gray=127)
    
    if not cfg.delight_model_dir.exists():
        return reference
    
    try:
        from hy3dgen.texgen.utils.dehighlight_utils import Light_Shadow_Remover
        
        delight_cfg = SimpleNamespace(
            device=cfg.device,
            light_remover_ckpt_path=str(cfg.delight_model_dir),
        )
        delight = Light_Shadow_Remover(delight_cfg)
        return delight(reference)
    except ImportError:
        return reference
```

**Decisions:**
- **Gray background** (Research 2: neutral starting point for paint model)
- **Optional** — gracefully degrades if delight model not downloaded
- Import from `hy3dgen` (Hunyuan3D-2.0 codebase) — requires the v2.0 repo cloned
- Falls back to plain gray composite if import fails

---

### Step K: Run Multiview Paint Diffusion

**Purpose:** Generate 6 albedo views + 6 metallic-roughness views conditioned on geometry + appearance.

**Implementation:**
```python
# src/paint_multiview.py

from diffusers import EulerAncestralDiscreteScheduler
from omegaconf import OmegaConf

class MultiviewDiffusionNet:
    """
    Standalone multiview paint model loader.
    
    Based on Research 1's LocalMultiviewDiffusionNet with:
    - Euler Ancestral scheduler (trailing timestep spacing)
    - Conditional DINO feature injection (gated on unet.use_dino)
    - PBR mode detection (albedo + metallic-roughness output split)
    - VAE slicing/tiling for memory efficiency
    """
    
    def __init__(self, cfg):
        from hunyuanpaintpbr.pipeline import HunyuanPaintPipeline
        
        self.device = cfg.device
        model_dir = cfg.models_dir / "hunyuan3d-paintpbr-v2-1"
        paint_cfg_path = cfg.third_party_dir / "Hunyuan3D-2.1" / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"
        
        self.paint_cfg = OmegaConf.load(str(paint_cfg_path))
        self.mode = self.paint_cfg.model.params.stable_diffusion_config.custom_pipeline[2:]
        
        # Load pipeline
        self.pipeline = HunyuanPaintPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
        )
        
        # Swap scheduler to Euler Ancestral with trailing spacing
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            timestep_spacing="trailing",
        )
        
        # Memory optimizations
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.eval()
        
        setattr(self.pipeline, "view_size", 
                self.paint_cfg.model.params.get("view_size", 320))
        self.pipeline = self.pipeline.to(cfg.device)
        
        # Conditional DINO loading
        self.dino = None
        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            dino_dir = cfg.models_dir / "dinov2-giant"
            if not dino_dir.exists():
                raise FileNotFoundError(
                    "Paint UNet requires DINOv2 but models/dinov2-giant/ is missing"
                )
            self.dino = Dino_v2(str(dino_dir))
            self.dino = self.dino.to(device=cfg.device, dtype=torch.float16)
    
    @torch.inference_mode()
    def __call__(self, reference_rgb, conditions, cfg):
        """
        Run multiview texture diffusion.
        
        Args:
            reference_rgb: PIL.Image (RGB, 512x512) — appearance anchor
            conditions: List[PIL.Image] — [normals...] + [positions...]
            cfg: pipeline config with paint_steps, paint_guidance_scale, seed
        
        Returns:
            dict with 'albedo': List[PIL.Image] and 'mr': List[PIL.Image]
        """
        input_img = reference_rgb.resize(
            (cfg.view_size, cfg.view_size)
        ).convert("RGB")
        
        control_images = [
            img.resize((cfg.view_size, cfg.view_size)) for img in conditions
        ]
        
        num_view = len(control_images) // 2
        normal_images = [[control_images[i] for i in range(num_view)]]
        position_images = [[control_images[i + num_view] for i in range(num_view)]]
        
        kwargs = {
            "width": cfg.view_size,
            "height": cfg.view_size,
            "num_in_batch": num_view,
            "images_normal": normal_images,
            "images_position": position_images,
            "generator": torch.Generator(device=self.device).manual_seed(cfg.seed),
        }
        
        if self.dino is not None:
            kwargs["dino_hidden_states"] = self.dino(input_img)
        
        outputs = self.pipeline(
            [input_img],
            num_inference_steps=cfg.paint_steps,
            prompt="high quality",
            sync_condition=None,
            guidance_scale=cfg.paint_guidance_scale,
            **kwargs,
        ).images
        
        if "pbr" in self.mode:
            return {
                "albedo": outputs[:num_view],
                "mr": outputs[num_view:],
            }
        return {"albedo": outputs, "mr": outputs}
```

**Decisions:**
- **Full reimplementation** of multiview diffusion loading (Research 1: `LocalMultiviewDiffusionNet`)
- **Euler Ancestral scheduler** with `timestep_spacing="trailing"` (verified from wrapper source)
- **DINO gating** on `unet.use_dino` (Research 1: only load if needed)
- **VAE slicing + tiling** for memory efficiency (Research 2: documented as key optimization)
- **PBR mode detection** from config — split output into albedo + MR
- **Pass `num_steps`, `guidance_scale`, `seed`** directly (Research 1 + 2 agree)
- First image is the DINO anchor (confirmed by wrapper source: `input_images[0:1]`)

---

### Step L: Upscale Generated Views

**Purpose:** Increase resolution of multiview images before baking for higher texture quality.

**Implementation:**
```python
# src/paint_multiview.py

def upscale_views(images: list, target_size: int, use_realesrgan: bool = False) -> list:
    """
    Upscale multiview images to render_size for baking.
    
    Two modes:
    - Lanczos resize (fast, acceptable quality)
    - RealESRGAN 4x upscale (slower, higher quality)
    """
    if not images:
        return images
    
    if use_realesrgan:
        from utils.image_super_utils import imageSuperNet
        # Load RealESRGAN model
        super_cfg = SimpleNamespace(
            realesrgan_ckpt_path="models/RealESRGAN_x4plus.pth"
        )
        super_model = imageSuperNet(super_cfg)
        return [super_model(img) for img in images]
    else:
        return [
            img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            for img in images
        ]
```

**Decisions:**
- Default to Lanczos resize (fast, deterministic)
- Optional RealESRGAN path for maximum quality (upstream monolithic pipeline uses this)
- Target `render_size` (typically 2x `texture_size`)

---

### Step M: Bake into UV Textures

**Purpose:** Back-project multiview renders into UV texture space with cosine-weighted blending.

**Implementation:**
```python
# src/bake_texture.py

from typing import List, Tuple
from PIL import Image

def bake_multiview_textures(
    pipeline: PaintPipeline,
    albedo_views: List[Image.Image],
    mr_views: List[Image.Image],
    cfg,
) -> Tuple:
    """
    Bake multiview images into UV texture maps.
    
    Algorithm per UV texel:
    1. Determine which camera views can see this texel (visibility test)
    2. For each visible view, sample color from the rendered image at the projected position
    3. Compute blend weight = base_weight × cos(angle)^bake_exp
    4. Blend all contributions; record coverage in trust mask
    
    Returns:
        (texture_albedo, mask_albedo, texture_mr, mask_mr)
        - textures: torch.Tensor [H, W, 3] in [0, 1]
        - masks: torch.Tensor [H, W, 1] — 1.0 = covered, 0.0 = needs inpaint
    """
    # Resize views to render_size for baking precision
    render_size = cfg.render_size
    albedo_resized = [img.resize((render_size, render_size), Image.Resampling.LANCZOS) 
                      for img in albedo_views]
    mr_resized = [img.resize((render_size, render_size), Image.Resampling.LANCZOS) 
                  for img in mr_views]
    
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
    
    return texture_albedo, mask_albedo, texture_mr, mask_mr
```

**Key insight:** The mask output is a **trust map** (`mask > 1e-8` = covered), not a segmentation mask. UV texels with mask=0 are occluded from all views and require inpainting.

---

### Step N: Inpaint Missing Texels

**Purpose:** Fill uncovered UV regions using two-pass inpainting.

**Implementation:**
```python
# src/inpaint_texture.py

import numpy as np

def inpaint_textures(
    pipeline: PaintPipeline,
    texture_albedo,
    mask_albedo,
    texture_mr,
    mask_mr,
    vertex_inpaint: bool = True,
    method: str = "NS",
):
    """
    Two-pass inpainting:
    
    Pass 1 (vertex_inpaint=True):
        Uses compiled meshVerticeInpaint C++ extension to propagate colors
        from covered vertices to nearby uncovered vertices along mesh edges.
        Preserves geometric continuity and reduces visible seams.
    
    Pass 2 (cv2.inpaint):
        Standard OpenCV inpainting fills remaining 2D gaps in UV space.
        Methods: "NS" = Navier-Stokes, "TELEA" = Fast Marching
    
    If meshVerticeInpaint extension is not compiled, falls back to cv2-only.
    """
    # Convert mask to uint8 for cv2
    def mask_to_uint8(mask):
        if hasattr(mask, 'squeeze'):
            arr = mask.squeeze(-1).detach().float().cpu().numpy()
        else:
            arr = np.asarray(mask).squeeze()
        return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    
    mask_albedo_np = mask_to_uint8(mask_albedo)
    mask_mr_np = mask_to_uint8(mask_mr)
    
    with torch.inference_mode():
        refined_albedo = pipeline.view_processor.texture_inpaint(
            texture_albedo, mask_albedo_np, vertex_inpaint, method
        )
        refined_mr = pipeline.view_processor.texture_inpaint(
            texture_mr, mask_mr_np, vertex_inpaint, method
        )
    
    return refined_albedo, refined_mr
```

**Decisions:**
- **Always try vertex-aware inpaint first** — produces much better seam quality
- **"NS" (Navier-Stokes)** as default method (Research 1: matches wrapper default)
- Graceful fallback if `meshVerticeInpaint` not compiled

---

### Step O: Apply Textures and Export GLB

**Purpose:** Set final textures on the mesh and export with PBR materials.

**Implementation:**
```python
# src/export_glb.py

from pathlib import Path

def export_textured_mesh(
    pipeline: PaintPipeline,
    refined_albedo,
    refined_mr,
    mesh,
    output_path: Path,
) -> Path:
    """
    Apply inpainted textures and export as GLB with PBR materials.
    
    Uses create_glb_with_pbr_materials (NOT convert_obj_to_glb which doesn't exist).
    The function expects texture files next to the OBJ:
    - mesh.jpg (albedo)
    - mesh_metallic.jpg
    - mesh_roughness.jpg
    """
    from convert_utils import create_glb_with_pbr_materials
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_obj = output_path.with_suffix(".obj")
    output_glb = output_path.with_suffix(".glb")
    
    # Load mesh and apply textures
    pipeline.load_mesh(mesh)
    pipeline.render.set_texture(refined_albedo, force_set=True)
    pipeline.render.set_texture_mr(refined_mr)
    
    # Save OBJ (creates texture files alongside)
    pipeline.render.save_mesh(str(output_obj), downsample=False)
    
    # Convert to GLB with PBR materials
    stem = output_obj.stem
    textures = {
        "albedo": str(output_obj.with_suffix(".jpg")),
        "metallic": str(output_obj.with_name(f"{stem}_metallic.jpg")),
        "roughness": str(output_obj.with_name(f"{stem}_roughness.jpg")),
    }
    create_glb_with_pbr_materials(str(output_obj), textures, str(output_glb))
    
    return output_glb
```

**Decisions:**
- **`create_glb_with_pbr_materials`** — the correct function name from `convert_utils.py` (Research 1: Tutorial's `convert_obj_to_glb` doesn't exist)
- **Explicit texture file paths** — the GLB builder needs to find albedo, metallic, roughness JPEGs
- **`downsample=False`** — preserve full mesh resolution in output (Research 1 finding)
- **`force_set=True`** on `set_texture` — overrides any existing texture data

---

## 7. Configuration Object

```python
# src/config.py

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class CameraConfig:
    azimuths: List[int] = field(default_factory=lambda: [0, 90, 180, 270, 0, 180])
    elevations: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 90, -90])
    weights: List[float] = field(default_factory=lambda: [1.0, 0.1, 0.5, 0.1, 0.05, 0.05])
    ortho_scale: float = 1.0

@dataclass
class PipelineConfig:
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    models_dir: Path = field(init=False)
    third_party_dir: Path = field(init=False)
    delight_model_dir: Path = field(init=False)
    
    # Device
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    
    # Shape generation
    shape_steps: int = 50
    shape_guidance_scale: float = 5.0
    octree_resolution: int = 384
    use_multiview_shape: bool = False
    shape_model_subfolder: str = "hunyuan3d-dit-v2-mv"
    
    # Mesh processing
    target_faces: int = 200_000
    normalize_mesh: bool = True
    
    # Camera
    camera: CameraConfig = field(default_factory=CameraConfig)
    
    # Paint / texture
    view_size: int = 512
    render_size: int = 2048
    texture_size: int = 4096
    paint_steps: int = 10
    paint_guidance_scale: float = 3.0
    use_delight: bool = True
    use_realesrgan: bool = False
    
    # Inpainting
    vertex_inpaint: bool = True
    inpaint_method: str = "NS"
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        self.models_dir = self.project_root / "models"
        self.third_party_dir = self.project_root / "third_party"
        self.delight_model_dir = self.models_dir / "delight"
```

---

## 8. VRAM Budget

| Stage | Peak VRAM | Can Be Offloaded After |
|-------|-----------|----------------------|
| Shape generation (DiT) | ~10 GB | Yes — `del pipeline; torch.cuda.empty_cache()` |
| Paint pipeline (multiview diffusion) | ~21 GB | Yes — `del` models after export |
| **Sequential total** | **~21 GB** | Run shape first, free, then paint |
| **Concurrent total** | **~29 GB** | Not recommended |

**Memory optimization priority:**
1. Run shape and paint sequentially (never concurrently)
2. Enable VAE slicing + tiling on paint pipeline
3. Reduce `texture_size` from 4096 to 2048 (biggest single VRAM win)
4. Reduce `render_size` from 2048 to 1024
5. Use model CPU offload if still tight

---

## 9. CLI Interface

```bash
# Single-image mode (Path A)
python main.py \
    --image inputs/front.png \
    --output outputs/model.glb \
    --shape-steps 50 \
    --paint-steps 10 \
    --guidance-scale 5.0 \
    --paint-guidance 3.0 \
    --texture-size 1024 \
    --view-size 512 \
    --seed 42

# Four-view mode (Path B)
python main.py \
    --front inputs/front.png \
    --left inputs/left.png \
    --right inputs/right.png \
    --back inputs/back.png \
    --output outputs/model.glb \
    --use-multiview-shape \
    --shape-steps 30 \
    --paint-steps 10 \
    --seed 1234

# Mesh-only (skip texturing)
python main.py \
    --image inputs/front.png \
    --output outputs/model.glb \
    --no-texture

# Debug mode (save all intermediates)
python main.py \
    --image inputs/front.png \
    --output outputs/model.glb \
    --save-intermediates
```

---

## 10. Intermediate Output Structure

```
outputs/model/
├── 01_preprocessed/
│   ├── front.png              # Input after background removal
│   └── front_alpha.png        # Alpha channel
├── 02_mesh/
│   ├── mesh_raw.glb           # Direct from shape pipeline
│   ├── mesh_postprocessed.glb # After cleanup + decimation
│   └── mesh_uv.glb            # After UV unwrap
├── 03_normal_maps/
│   ├── normal_00.png          # Front view normal map
│   ├── normal_01.png          # Right view
│   └── ...
├── 04_position_maps/
│   ├── position_00.png
│   └── ...
├── 05_reference/
│   └── reference_delighted.png  # After delight processing
├── 06_mv_albedo/
│   ├── albedo_00.png          # Front-view generated albedo
│   └── ...
├── 07_mv_mr/
│   ├── mr_00.png              # Front-view metallic-roughness
│   └── ...
├── 08_bake/
│   ├── baked_albedo.png       # UV texture after baking
│   ├── baked_albedo_mask.png  # Trust mask
│   ├── baked_mr.png
│   └── baked_mr_mask.png
└── 09_refine/
    ├── refined_albedo.png     # After inpainting
    └── refined_mr.png
```

---

## 11. Key Technical Constraints

| Constraint | Reason | Consequence of Violation |
|-----------|--------|------------------------|
| UV unwrap BEFORE rendering | Renderer projects into UV coordinates | Black textures / empty bake |
| View order consistent across render→paint→bake | Multiview model indexes by position | Mirrored/shifted textures |
| `ortho_scale` passed to `MeshRender` | Controls camera frustum | Wrong projection scale → clipped geometry |
| Front view = appearance anchor | DINO features computed from `input_images[0]` | Wrong colors if non-front used |
| `render_size` >= `texture_size` for baking | Bake samples from rendered images | Aliased/blurry UV textures |
| Texture files next to OBJ for GLB export | `create_glb_with_pbr_materials` reads by filename | GLB appears untextured |
| Same mesh instance loaded into renderer before bake AND before export | Renderer's internal state must match | UV mismatch between bake and export |

---

## 12. Validation Checklist

```bash
# 1. CUDA works
python -c "import torch; assert torch.cuda.is_available(); print(f'OK: {torch.cuda.get_device_name(0)}')"

# 2. Extensions compiled
python -c "import custom_rasterizer_kernel; print('custom_rasterizer: OK')"
python -c "import sys; sys.path.insert(0, 'third_party/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer'); import mesh_inpaint_processor; print('mesh_inpaint: OK')"

# 3. Models downloaded
ls models/hunyuan3d-dit-v2-1/model.fp16.ckpt
ls models/hunyuan3d-vae-v2-1/model.fp16.ckpt
ls models/hunyuan3d-paintpbr-v2-1/
ls models/dinov2-giant/

# 4. Shape pipeline loads
python -c "
import sys; sys.path.insert(0, 'third_party/Hunyuan3D-2.1/hy3dshape')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
p = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1', subfolder='hunyuan3d-dit-v2-1')
print(f'Shape pipeline OK. Device: {p.device}')
del p; import torch; torch.cuda.empty_cache()
"

# 5. xatlas version
python -c "import xatlas; print(f'xatlas version: {xatlas.__version__}')"
# Should be 0.0.10

# 6. End-to-end run
python main.py --image inputs/test.png --output outputs/test.glb --save-intermediates
```

---

## 13. Summary of Decisions and Sources

| Decision | Chosen Approach | Source | Why |
|----------|----------------|--------|-----|
| API surface | `from_pretrained` | Research 2 | Stable public API |
| Mesh output | `output_type="trimesh"` | Research 2 | Avoids unstable latent split |
| Background for paint | Gray (127) | Research 2 | Reduces lighting bias |
| Delight step | Optional `Light_Shadow_Remover` | Research 2 | Quality improvement for photos |
| `ortho_scale` to MeshRender | Yes | Research 1 + 2 | Critical rendering parameter |
| GLB export function | `create_glb_with_pbr_materials` | Research 1 | Correct function name |
| Paint model init | `LocalMultiviewDiffusionNet` pattern | Research 1 | Full control over scheduler/DINO |
| xatlas version | 0.0.10 | Research 2 | Official 2.1 pin |
| Inference context | `torch.inference_mode()` | Research 2 | More efficient than no_grad |
| Face target | 200,000 | Research 1 + 2 | Better texture quality than 40k |
| Mesh normalization | Yes | Research 2 | Consistent renderer behavior |
| VAE slicing/tiling | Yes | Research 1 + 2 | Memory optimization |
| Camera config | Dataclass | Research 1 | Type safety |
| File architecture | Multi-file `src/` package | Research 2 | Maintainability |
| Intermediate saving | Numbered stage dirs | Research 2 | Clearer debugging |
| Error handling | `FileNotFoundError` + validation | Research 1 | Production robustness |
| Theoretical docs | Tutorial explanations | Tutorial | Best depth on camera/baking theory |
