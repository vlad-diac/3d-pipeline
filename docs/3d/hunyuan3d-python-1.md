# Standalone Hunyuan3D-2.1 Python Script Tutorial

The cleanest standalone way to reproduce the working entity["organization","ComfyUI-Hunyuan3d-2-1","github repo"] flow is **not** to import `nodes.py` directly. The active path in that wrapper is: single-image shape diffusion → explicit VAE decode to mesh → mesh postprocess → UV unwrap → user-defined camera config → split multiview paint generation → UV baking → inpaint → export. The wrapper’s true front/left/back/right **multiview geometry** path is only present as commented or dormant code, and it references a `dit_config_2_1_mv.yaml` file that is not actually present in the repo’s `configs/` directory. That means the reliable production path today is **single-image geometry + multiview texture generation**, not true multiview geometry generation. citeturn10view0turn10view1turn10view2turn29view1turn28view1

The tutorial below keeps the **same stage ordering and same underlying implementation calls** as the wrapper, but turns them into a regular Python project. For geometry, it uses the upstream shape pipeline and VAE directly because that is much less brittle outside ComfyUI. For texturing, it reproduces the wrapper’s **split** paint flow rather than the upstream monolithic `Hunyuan3DPaintPipeline.__call__`, because the wrapper intentionally returns generated multiview `albedo` and `mr` images first, then exposes separate bake/inpaint/export stages. citeturn33view1turn35view1turn22view0turn23view0

## Environment Setup

Upstream Hunyuan3D-2.1 says its tested stack is **Python 3.10 with PyTorch 2.5.1+cu124**. The official README also says you should compile the `custom_rasterizer` extension and the `mesh_inpaint_processor` extension for the paint stack. By contrast, the visualbruno wrapper README says that its ComfyUI packaging was tested on **Windows 11, Python 3.12, Torch >= 2.6.0 + cu126** and even ships prebuilt wheels for Windows. For a standalone Linux or RunPod workflow, the least-friction choice is the **official upstream stack**: Python 3.10, Torch 2.5.1 cu124, and source builds for the two extensions. citeturn12view2turn42view0turn42view1turn44search1

The official repo’s model zoo says shape generation takes about **10 GB VRAM**, texture generation about **21 GB VRAM**, and running both together around **29 GB VRAM**. A community RunPod template in the repo specifically calls out the **A40** as a good choice. That lines up well with a single-GPU Linux setup for the full pipeline. citeturn12view1turn40view1

The wrapper and official repos do **not** publish a strict inference lockfile. The pinset below is therefore a **standalone reproducibility layer** centered on the upstream-tested torch/CUDA pair and the packages the two repos actually import. `xformers` is optional here; it is **not required** by the active wrapper flow, and the wrapper’s shape loader defaults to `attention_mode="sdpa"`. If you want an xformers wheel aligned with PyTorch 2.5.1, `xformers==0.0.28.post3` is the matching release. citeturn13view0turn14view0turn34view2turn43search0

### `setup.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

APT_PREFIX=""
if command -v sudo >/dev/null 2>&1; then
  APT_PREFIX="sudo"
fi

if command -v apt-get >/dev/null 2>&1; then
  ${APT_PREFIX} apt-get update || true
  ${APT_PREFIX} apt-get install -y \
    build-essential \
    git \
    git-lfs \
    wget \
    curl \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 || true
fi

if [[ ! -d "$ROOT/conda" ]]; then
  curl -L -o Miniforge3.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3.sh -b -p "$ROOT/conda"
fi

source "$ROOT/conda/etc/profile.d/conda.sh"

if ! conda env list | grep -q "hunyuan3d-standalone"; then
  conda create -y -n hunyuan3d-standalone python=3.10
fi

conda activate hunyuan3d-standalone

python -m pip install --upgrade pip setuptools wheel

# Official tested torch stack
python -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

# Optional; safe to skip if the wheel is unavailable for your platform.
python -m pip install \
  xformers==0.0.28.post3 \
  --index-url https://download.pytorch.org/whl/cu124 || true

python -m pip install -r requirements.txt

mkdir -p third_party

if [[ ! -d third_party/Hunyuan3D-2.1 ]]; then
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 third_party/Hunyuan3D-2.1
fi

if [[ ! -d third_party/ComfyUI-Hunyuan3d-2-1 ]]; then
  git clone https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1 third_party/ComfyUI-Hunyuan3d-2-1
fi

pushd third_party/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
python -m pip install -e .
popd

pushd third_party/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
popd

mkdir -p inputs outputs models scripts

python scripts/download_models.py

echo
echo "Done."
echo "Activate with:"
echo "  source \"$ROOT/conda/etc/profile.d/conda.sh\""
echo "  conda activate hunyuan3d-standalone"
```

### `requirements.txt`

```txt
accelerate==1.1.1
basicsr==1.4.2
configargparse==1.7
cupy-cuda12x==13.3.0
diffusers==0.31.0
einops==0.8.0
huggingface_hub==0.26.3
omegaconf==2.3.0
onnxruntime==1.19.2
opencv-python==4.10.0.84
open3d==0.18.0
pillow==10.4.0
pybind11==2.13.6
pygltflib==1.16.2
pymeshlab==2023.12.post2
pytorch-lightning==2.4.0
pyyaml==6.0.2
realesrgan==0.3.0
rembg==2.0.59
safetensors==0.4.5
scikit-image==0.24.0
timm==1.0.11
trimesh==4.4.9
xatlas==0.0.9
```

### CUDA and Linux notes

The `custom_rasterizer` extension builds a CUDA module from `rasterizer.cpp`, `grid_neighbor.cpp`, and `rasterizer_gpu.cu`; the mesh inpaint extension is built by compiling `mesh_inpaint_processor.cpp` with `pybind11`. On Linux or RunPod, make sure `nvcc` and the host compiler are available before you run `setup.sh`. If either extension fails to build, the texture stage will not run correctly even if shape generation works. citeturn42view0turn42view1turn25view6

## Repository, Models, and Project Layout

For the standalone project, the **official repo** is your runtime dependency source, and the **visualbruno repo** is your architectural reference for the split-node flow. The files you actually need from the wrapper repo are:

- `nodes.py` for the stage mapping.
- `configs/dit_config_2_1.yaml` for the wrapper’s shape single-file loader reference.
- `hy3dpaint/textureGenPipeline.py` for the split paint flow.
- `hy3dpaint/utils/pipeline_utils.py` for `bake_from_multiview` and `texture_inpaint`.
- `hy3dpaint/utils/multiview_utils.py` for multiview diffusion loading and control-image packing.
- `hy3dpaint/utils/uvwrap_utils.py` for xatlas UV unwrap.
- `hy3dpaint/DifferentiableRenderer/*` and `hy3dpaint/custom_rasterizer/*` for the renderer and custom extension sources. citeturn44search1turn22view0turn23view0turn23view1turn24view3turn42view0turn42view1

For model files, a standalone local `models/` folder is easier to reason about than relying on cache-only resolution. The important downloads are:

- `models/hunyuan3d-dit-v2-1/config.yaml`
- `models/hunyuan3d-dit-v2-1/model.fp16.ckpt`
- `models/hunyuan3d-vae-v2-1/model.fp16.ckpt`
- `models/hunyuan3d-paintpbr-v2-1/` as the full diffusers paint pipeline folder
- `models/dinov2-giant/` if the paint UNet uses DINO features

The official HF tree shows the DiT folder contains `config.yaml` and a `model.fp16.ckpt` file of about **7.37 GB**. The VAE lives under a separate `hunyuan3d-vae-v2-1/` folder, where `model.fp16.ckpt` is about **656 MB**. The paint stack loads `hunyuan3d-paintpbr-v2-1/*`, and both the official and wrapper paint configs default the DINO checkpoint path to `facebook/dinov2-giant`. The wrapper README also documents the ComfyUI-style checkpoint naming convention `hunyuan3d-dit-v2-1.ckpt` and `hunyuan3d-vae-v2-1.ckpt`, but the standalone project below uses the official HF subfolder layout instead. citeturn45search1turn45search0turn44search8turn23view1turn17view0turn44search1

### Project tree

```text
your-project/
├── setup.sh
├── requirements.txt
├── run_hunyuan3d.py
├── scripts/
│   └── download_models.py
├── inputs/
├── outputs/
├── models/
│   ├── hunyuan3d-dit-v2-1/
│   │   ├── config.yaml
│   │   └── model.fp16.ckpt
│   ├── hunyuan3d-vae-v2-1/
│   │   └── model.fp16.ckpt
│   ├── hunyuan3d-paintpbr-v2-1/
│   └── dinov2-giant/
└── third_party/
    ├── Hunyuan3D-2.1/
    └── ComfyUI-Hunyuan3d-2-1/
```

### `scripts/download_models.py`

```python
#!/usr/bin/env python3
from pathlib import Path
import os
import urllib.request

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

def download_repo_subfolder(repo_id: str, allow_patterns, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )

# Hunyuan3D-2.1 DiT
download_repo_subfolder(
    repo_id="tencent/Hunyuan3D-2.1",
    allow_patterns=["hunyuan3d-dit-v2-1/*"],
    local_dir=MODELS,
)

# Hunyuan3D-2.1 VAE
download_repo_subfolder(
    repo_id="tencent/Hunyuan3D-2.1",
    allow_patterns=["hunyuan3d-vae-v2-1/*"],
    local_dir=MODELS,
)

# Hunyuan3D paint diffusers folder
download_repo_subfolder(
    repo_id="tencent/Hunyuan3D-2.1",
    allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
    local_dir=MODELS,
)

# DINOv2 giant
snapshot_download(
    repo_id="facebook/dinov2-giant",
    local_dir=str(MODELS / "dinov2-giant"),
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)

# Optional RealESRGAN weight, only needed if you later restore the upstream monolithic paint enhance stage.
realesrgan_path = MODELS / "RealESRGAN_x4plus.pth"
if not realesrgan_path.exists():
    urllib.request.urlretrieve(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        realesrgan_path,
    )

print("Model download completed under:", MODELS)
```

If the HF download fails with 401 or rate-limit errors, authenticate first with a token, for example by exporting `HF_TOKEN` or logging in with the HF CLI before running the download script. The code paths above are all consistent with how the official and wrapper paint stacks resolve model folders. citeturn23view1turn45search1turn44search0

## Python Script Architecture

The wrapper’s active graph maps almost one-to-one to a plain Python program:

- `Hy3DMeshGenerator` → load shape diffusion and return a latent instead of a mesh.
- `Hy3D21VAELoader` → instantiate `ShapeVAE` with the wrapper’s exact config values.
- `Hy3D21VAEDecode` → `vae.decode(...)` then `vae.latents2mesh(...)`.
- `Hy3D21PostprocessMesh` → connected-component cleanup, degenerate-face removal, optional reduction.
- `Hy3D21MeshUVWrap` → `xatlas.parametrize(...)`.
- `Hy3D21CameraConfig` → fixed azimuth/elevation/weight lists plus `ortho_scale`.
- `Hy3DMultiViewsGenerator` → render normal/position maps + call the paint diffusion model.
- `Hy3DBakeMultiViews` → cosine-weighted view back-projection into UV textures.
- `Hy3DInPaint` → vertex-aware hole fill first, OpenCV Navier-Stokes second.
- `Hy3D21ExportMesh` → save textured OBJ and convert to GLB with PBR textures. citeturn10view0turn10view2turn22view0turn23view0turn24view3turn25view6

One important practical change from the upstream official paint pipeline: the **wrapper split flow exits early after multiview image generation** and exposes bake/inpaint/export as separate callable stages. The official monolithic paint pipeline would otherwise go on to enhance, bake, inpaint, and save inside one `__call__`. That is exactly why the standalone script below implements a local `StandalonePaintPipeline` class instead of just calling the official paint pipeline end-to-end. citeturn18view3turn18view4turn22view0

### Step-by-step implementation map

The full code later in this report is the exact runnable implementation. This stage map tells you what each function does and which wrapper node it replaces.

### Input loading and preprocessing

- **A. load input image/images**
  - **Script functions:** `choose_primary_image`, `load_image_rgba`, `collect_multiview_input_dict`
  - **Node mapping:** pre-node image handling before `Hy3DMeshGenerator`
  - **Input → output:** CLI path(s) → `PIL.Image` in RGBA
  - **Failure cases:** missing file, unreadable PNG/JPEG, fully transparent image

- **B. preprocess image / alpha / background**
  - **Script functions:** `maybe_remove_background`, `rgba_to_rgb_white`
  - **Libraries/classes:** `PIL`, upstream `BackgroundRemover`
  - **Why:** shape stage is more stable when the object is centered and isolated, and the official shape image processor is configured for a 512 input with `border_ratio: 0.15`; the paint stage needs an RGB conditioning image, so RGBA gets composited onto white before texturing. citeturn28view0turn18view3
  - **Failure cases:** poor alpha mask, clipped object, too much background, subject touching image border

### Shape diffusion and VAE decode

- **C. load Hunyuan shape pipeline**
  - **Script function:** `load_shape_pipeline`
  - **Libraries/classes:** `Hunyuan3DDiTFlowMatchingPipeline.from_single_file`
  - **Why:** use the official shape pipeline outside ComfyUI while loading **local** `config.yaml` + `model.fp16.ckpt`
  - **Failure cases:** missing `models/hunyuan3d-dit-v2-1/config.yaml`, missing `.ckpt`, incompatible torch/CUDA

- **D. generate HY3D latent**
  - **Script function:** `generate_shape_latent`
  - **Libraries/classes:** upstream `Hunyuan3DDiTFlowMatchingPipeline.__call__(..., output_type="latent")`
  - **Wrapper mapping:** `Hy3DMeshGenerator` does the same conceptually, but its patched wrapper pipeline returns latents directly from `__call__` rather than exporting a mesh. citeturn35view1turn33view1
  - **Input → output:** `PIL.Image` → latent tensor
  - **Failure cases:** OOM during diffusion, bad subject framing, wrong checkpoint path

- **E. load ShapeVAE**
  - **Script function:** `load_shape_vae`
  - **Libraries/classes:** `ShapeVAE`
  - **Wrapper mapping:** `Hy3D21VAELoader`
  - **Why:** the wrapper node hardcodes the VAE config; the standalone script uses those same parameter values in `VAE_CONFIG`. citeturn9view3
  - **Failure cases:** VAE checkpoint missing, state-dict mismatch, wrong dtype/device

- **F. decode latent into trimesh**
  - **Script function:** `decode_latent_to_mesh`
  - **Libraries/classes:** `vae.decode`, `vae.latents2mesh`, `trimesh.Trimesh`
  - **Wrapper mapping:** `Hy3D21VAEDecode`
  - **Why:** the wrapper decodes the latent and then flips face winding before building a `trimesh` mesh
  - **Failure cases:** NaN latent, OOM during marching cubes, degenerate export result

### Mesh cleanup, UV wrap, and camera setup

- **G. postprocess mesh**
  - **Script function:** `postprocess_mesh`
  - **Libraries/classes:** `trimesh`, `pymeshlab`
  - **Wrapper mapping:** `Hy3D21PostprocessMesh`
  - **Input → output:** raw `trimesh.Trimesh` → cleaned `trimesh.Trimesh`
  - **Failure cases:** extremely fragmented mesh, huge face count, decimation breaking topology

- **H. UV unwrap mesh**
  - **Script function:** `maybe_uv_wrap`
  - **Libraries/classes:** `xatlas`
  - **Wrapper mapping:** `Hy3D21MeshUVWrap`
  - **Why:** the wrapper’s UV step is exactly `xatlas.parametrize(mesh.vertices, mesh.faces)` and then rewrites vertices, faces, and `mesh.visual.uv`. citeturn24view3
  - **Failure cases:** xatlas crash on pathological meshes, too many faces, invalid topology

- **I. build camera config**
  - **Script function:** `build_camera_config`
  - **Wrapper mapping:** `Hy3D21CameraConfig`
  - **Default values:** azimuths `[0, 90, 180, 270, 0, 180]`, elevations `[0, 0, 0, 0, 90, -90]`, weights `[1, 0.1, 0.5, 0.1, 0.05, 0.05]`, `ortho_scale=1.0` in the wrapper split flow. citeturn10view4turn22view0
  - **Failure cases:** list length mismatch, extreme orthographic scale, over-trusting side views

### Paint model, baking, inpaint, and export

- **J. initialize Hunyuan3DPaintPipeline**
  - **Script class:** `StandalonePaintPipeline`
  - **Underlying classes:** `MeshRender`, `ViewProcessor`, `HunyuanPaintPipeline`, `Dino_v2`
  - **Why:** reproduce visualbruno’s split `Hunyuan3DPaintConfig/Hunyuan3DPaintPipeline` without ComfyUI-only imports
  - **Failure cases:** missing compiled rasterizer, missing DINO folder, import-path mistakes

- **K. render normal and position maps**
  - **Script method:** `StandalonePaintPipeline.generate_multiviews`
  - **Underlying calls:** `render_normal_multiview`, `render_position_multiview`
  - **Wrapper mapping:** `Hy3DMultiViewsGenerator`
  - **Failure cases:** renderer extension not compiled, black images from invalid UV or broken camera config

- **L. run multiview diffusion paint model**
  - **Script class/method:** `LocalMultiviewDiffusionNet.__call__`
  - **Underlying behavior:** pack `normal_maps` and `position_maps` as control inputs, then call `HunyuanPaintPipeline(...).images`
  - **Why:** this mirrors the wrapper’s `multiviewDiffusionNet.forward_one(...)` and keeps the stage split visible. In the wrapper raw code, the model is loaded from `hunyuan3d-paintpbr-v2-1/*`, the scheduler is swapped to Euler ancestral, and DINO is only added if `unet.use_dino` is true. citeturn23view1
  - **Failure cases:** HF/local model mismatch, DINO missing, OOM at 768 or large texture sizes

- **M. bake albedo and metallic-roughness textures**
  - **Script method:** `StandalonePaintPipeline.bake_from_multiview`
  - **Underlying calls:** `ViewProcessor.bake_from_multiview`
  - **Wrapper mapping:** `Hy3DBakeMultiViews`
  - **Failure cases:** empty trust mask, heavily occluded surfaces, low-res view artifacts

- **N. inpaint missing UV regions**
  - **Script method:** `StandalonePaintPipeline.inpaint`
  - **Underlying calls:** `texture_inpaint` → mesh-aware vertex inpaint → `cv2.inpaint(..., cv2.INPAINT_NS)`
  - **Wrapper mapping:** `Hy3DInPaint`
  - **Failure cases:** `meshVerticeInpaint` extension missing, mask inversion mistakes, over-smoothed fill regions. citeturn25view6turn25view7

- **O. apply textures to mesh**
  - **Script methods:** `set_texture_albedo`, `set_texture_mr`
  - **Underlying calls:** `render.set_texture(..., force_set=True)` and `render.set_texture_mr(...)`
  - **Failure cases:** texture type mismatch, wrong channel order, uint8/float confusion

- **P. export GLB/OBJ**
  - **Script method:** `save_mesh`
  - **Underlying calls:** `render.save_mesh(...)` and `create_glb_with_pbr_materials(...)`
  - **Wrapper mapping:** `Hy3D21ExportMesh`
  - **Failure cases:** output path missing, broken `.mtl` references, bad GLB conversion if expected texture filenames are absent

## Multiview Geometry Variant

The dormant wrapper code for `Hy3D21MultiViewsMeshGenerator` shows the intended input contract clearly: it wanted a dictionary with keys `front`, `left`, `back`, and `right`, then called the shape pipeline with `image=view_dict`, `config_path="configs/dit_config_2_1_mv.yaml"`, and otherwise the same diffusion arguments. It also exposed the same `steps`, `guidance_scale`, and `seed` parameters. citeturn10view1turn35view1

The reason that variant is **not** runnable out of the box is twofold:

- The whole node block is commented/dormant in the wrapper. citeturn10view1
- The referenced `configs/dit_config_2_1_mv.yaml` is **absent** from the repo; the wrapper’s `configs/` folder contains `dit_config.yaml`, `dit_config_2_1.yaml`, and `dit_config_mini.yaml`, but no multiview 2.1 config. The raw URL for `dit_config_2_1_mv.yaml` also 404s. citeturn29view0turn29view1turn28view1

If you want to re-enable it, this is the practical patch plan:

```python
view_dict = {
    "front": Image.open(front_path).convert("RGBA"),
    "left": Image.open(left_path).convert("RGBA"),
    "back": Image.open(back_path).convert("RGBA"),
    "right": Image.open(right_path).convert("RGBA"),
}

latents = shape_pipeline(
    image=view_dict,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    generator=torch.Generator(device=device).manual_seed(seed),
    output_type="latent",
)
```

You then need all of the following:

- A real `dit_config_2_1_mv.yaml` whose `conditioner`, denoiser, and image preprocessing match whatever multiview conditioning the commented node expected.
- A checkpoint trained for that config.
- Confirmation that the shape image preprocessor and conditioner can still consume the dictionary input exactly as intended in the dormant path.

The most likely risk is that dictionary input support is only **partially wired**. The wrapper shape pipeline’s type annotation explicitly allows `dict` and `List[dict]`, which is promising, but without the missing config and active code path, you should treat this as **experimental** rather than production-ready. citeturn35view1turn10view1

## Camera and Texture Pipeline Details

The working split texturing path uses **six canonical cameras** by default: four cardinal views around the equator and two polar views. In the wrapper split config, the front view gets weight `1.0`, the back gets `0.5`, the left and right get `0.1`, and top and bottom get `0.05`. In practice, that means the pipeline prefers the image-facing front details but still gives nonzero authority to hidden or overhead surfaces during baking. citeturn10view4turn22view0

Normal maps and position maps are not optional decoration here. The paint model explicitly consumes them as the control bundle: the multiview generator concatenates `normal_maps + position_maps`, then splits those internally into `images_normal` and `images_position` before calling the diffusion model. That is the core condition signal that tells the paint model what 3D surface each generated view corresponds to. citeturn18view3turn23view1

The UV bake stage is **cosine-weighted back-projection**, not a simple overwrite. For each view, the renderer back-projects colors into UV space, raises the cosine visibility term to `bake_exp`, multiplies by the view weight, and then merges all projected textures in `fast_bake_texture`. The returned mask is a **trust map** or visibility-confidence map in UV space, which is why the wrapper thresholds it as `ori_trust_map > 1e-8`. It is not a semantic segmentation mask. citeturn23view0turn25view1

UV wrapping has to happen **before** baking because the renderer is literally projecting pixels into UV coordinates. No UVs means no canonical texture domain to accumulate into. That is also why the wrapper exposes `Hy3D21MeshUVWrap` before the texturing stages. citeturn24view3turn10view0

The inpaint stage is intentionally two-step. First, the renderer can call `meshVerticeInpaint(...)`, which uses vertex and UV topology to propagate plausible values into holes. Then it optionally finishes with `cv2.inpaint(..., cv2.INPAINT_NS)` to smooth remaining gaps. This is why missing the mesh inpaint extension often produces worse-looking or black UV islands even if OpenCV is installed. citeturn25view6turn25view7

## Full Runnable Code

The script below is the standalone deliverable. It expects:

- cloned official repo at `third_party/Hunyuan3D-2.1`
- models downloaded under `models/`
- compiled paint extensions already installed
- input image(s) in `inputs/`

It implements the active single-image geometry + split multiview texturing flow, exposes the required CLI, saves intermediate artifacts, and falls back safely when users provide front/left/back/right images for the currently-dormant multiview geometry path.

### `run_hunyuan3d.py`

```python
import argparse
import gc
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image


VAE_CONFIG = {
    "num_latents": 4096,
    "embed_dim": 64,
    "num_freqs": 8,
    "include_pi": False,
    "heads": 16,
    "width": 1024,
    "num_encoder_layers": 8,
    "num_decoder_layers": 16,
    "qkv_bias": False,
    "qk_norm": True,
    "scale_factor": 1.0039506158752403,
    "geo_decoder_mlp_expand_ratio": 4,
    "geo_decoder_downsample_ratio": 1,
    "geo_decoder_ln_post": True,
    "point_feats": 4,
    "pc_size": 81920,
    "pc_sharpedge_size": 0,
}

DEFAULT_CAMERA_AZIMS = [0, 90, 180, 270, 0, 180]
DEFAULT_CAMERA_ELEVS = [0, 0, 0, 0, 90, -90]
DEFAULT_VIEW_WEIGHTS = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def apply_torchvision_fix(official_repo: Path, logger: logging.Logger) -> None:
    fix_path = official_repo / "torchvision_fix.py"
    if not fix_path.exists():
        return
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("torchvision_fix", fix_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        if hasattr(module, "apply_fix"):
            module.apply_fix()
            logger.info("Applied torchvision_fix.py before importing Hunyuan modules.")
    except Exception as exc:
        logger.warning("torchvision_fix.py exists but could not be applied: %s", exc)


def add_repo_paths(official_repo: Path, logger: logging.Logger) -> None:
    hy3dshape_root = official_repo / "hy3dshape"
    hy3dpaint_root = official_repo / "hy3dpaint"
    if not hy3dshape_root.exists() or not hy3dpaint_root.exists():
        raise FileNotFoundError(
            f"Expected cloned official repo under {official_repo}, "
            f"with hy3dshape/ and hy3dpaint/ subdirectories."
        )
    apply_torchvision_fix(official_repo, logger)
    sys.path.insert(0, str(hy3dshape_root))
    sys.path.insert(0, str(hy3dpaint_root))


def rgba_to_rgb_white(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(white, image)
    return composed.convert("RGB")


def load_image_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def maybe_remove_background(image: Image.Image, remove_background: bool, background_remover=None) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if image.getbbox() is None:
        raise ValueError("Input image is fully transparent after loading.")
    if image.getchannel("A").getextrema() == (255, 255) and remove_background:
        if background_remover is None:
            return image
        return background_remover(image)
    return image


def save_pil_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_pil_list(folder: Path, images: Sequence[Image.Image], stem: str) -> List[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for index, image in enumerate(images):
        out_path = folder / f"{stem}_{index:02d}.png"
        image.save(out_path)
        paths.append(out_path)
    return paths


def tensor_hwcn_to_uint8(texture: torch.Tensor) -> np.ndarray:
    if not isinstance(texture, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(texture)!r}")
    tex = texture.detach().float().cpu().numpy()
    if tex.ndim != 3:
        raise ValueError(f"Expected HWC texture tensor, got shape {tex.shape}")
    tex = np.clip(tex, 0.0, 1.0)
    return (tex * 255).astype(np.uint8)


def save_texture_tensor(path: Path, texture: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_hwcn_to_uint8(texture)).save(path)


def mask_tensor_to_uint8(mask: torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.squeeze(-1).detach().float().cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255).astype(np.uint8)
    raise TypeError(f"Expected torch.Tensor mask, got {type(mask)!r}")


def save_mask_tensor(path: Path, mask: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_tensor_to_uint8(mask)).save(path)


def export_debug_mesh(path: Path, mesh: trimesh.Trimesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if not parts:
        return mesh
    best = max(parts, key=lambda m: len(m.faces))
    return best.copy()


def remove_degenerate_and_duplicate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh


def decimate_mesh_with_pymeshlab(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    import pymeshlab  # lazy import

    ms = pymeshlab.MeshSet()
    pm = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pm, "mesh")
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preservetopology=True,
        preserveboundary=True,
        preservenormal=True,
    )
    current = ms.current_mesh()
    out = trimesh.Trimesh(
        vertices=current.vertex_matrix(),
        faces=current.face_matrix(),
        process=False,
    )
    return out


def postprocess_mesh(
    mesh: trimesh.Trimesh,
    logger: logging.Logger,
    remove_floaters: bool = True,
    remove_degenerate_faces: bool = True,
    reduce_faces: bool = False,
    max_faces: int = 200000,
) -> trimesh.Trimesh:
    out = mesh.copy()
    if remove_floaters:
        out = keep_largest_component(out)
        logger.info("Removed floaters / kept largest connected component.")
    if remove_degenerate_faces:
        out = remove_degenerate_and_duplicate_faces(out)
        logger.info("Removed degenerate/duplicate faces and unreferenced vertices.")
    if reduce_faces and len(out.faces) > max_faces:
        out = decimate_mesh_with_pymeshlab(out, max_faces)
        logger.info("Reduced mesh to ~%d faces.", max_faces)
    return out


@dataclass
class CameraConfig:
    azims: List[int]
    elevs: List[int]
    weights: List[float]
    ortho_scale: float = 1.0


class LocalMultiviewDiffusionNet:
    def __init__(self, model_dir: Path, cfg_path: Path, dino_dir: Optional[Path], device: str, logger: logging.Logger):
        from diffusers import EulerAncestralDiscreteScheduler
        from omegaconf import OmegaConf
        from hunyuanpaintpbr.pipeline import HunyuanPaintPipeline

        self.device = device
        self.logger = logger
        self.cfg = OmegaConf.load(str(cfg_path))
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]
        self.pipeline = HunyuanPaintPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            timestep_spacing="trailing",
        )
        self.pipeline.set_progress_bar_config(disable=False)
        self.pipeline.eval()
        setattr(self.pipeline, "view_size", self.cfg.model.params.get("view_size", 320))
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline = self.pipeline.to(device)

        self.dino_v2 = None
        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            if dino_dir is None or not dino_dir.exists():
                raise FileNotFoundError(
                    "The paint UNet expects DINOv2 features, but the DINO model directory is missing. "
                    "Download facebook/dinov2-giant into models/dinov2-giant."
                )
            from hunyuanpaintpbr.unet.modules import Dino_v2

            self.dino_v2 = Dino_v2(str(dino_dir))
            self.dino_v2 = self.dino_v2.to(device=device, dtype=torch.float16 if str(device).startswith("cuda") else torch.float32)
            self.logger.info("Loaded DINOv2 from %s", dino_dir)

    @torch.no_grad()
    def __call__(
        self,
        images: List[Image.Image],
        conditions: List[Image.Image],
        prompt: str,
        custom_view_size: int,
        resize_input: bool,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Dict[str, List[Image.Image]]:
        seed_everything(seed)

        if not isinstance(images, list):
            images = [images]

        if resize_input:
            input_images = [img.resize((custom_view_size, custom_view_size)).convert("RGB") for img in images]
        else:
            input_images = [img.resize((self.pipeline.view_size, self.pipeline.view_size)).convert("RGB") for img in images]

        control_images = [img.resize((custom_view_size, custom_view_size)) for img in conditions]
        for idx, img in enumerate(control_images):
            if img.mode == "L":
                control_images[idx] = img.point(lambda x: 255 if x > 1 else 0, mode="1")

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs = {
            "width": custom_view_size,
            "height": custom_view_size,
            "num_in_batch": num_view,
            "images_normal": normal_image,
            "images_position": position_image,
            "generator": torch.Generator(device=self.device if str(self.device).startswith("cuda") else "cpu").manual_seed(seed),
        }

        if self.dino_v2 is not None:
            kwargs["dino_hidden_states"] = self.dino_v2(input_images[0])

        outputs = self.pipeline(
            input_images[0:1],
            num_inference_steps=num_steps,
            prompt=prompt,
            sync_condition=None,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            return {
                "albedo": outputs[:num_view],
                "mr": outputs[num_view:],
            }
        return {"hdr": outputs}


class StandalonePaintPipeline:
    def __init__(
        self,
        models_dir: Path,
        official_repo: Path,
        camera_config: CameraConfig,
        texture_size: int,
        view_size: int,
        device: str,
        logger: logging.Logger,
    ) -> None:
        from DifferentiableRenderer.MeshRender import MeshRender
        from utils.pipeline_utils import ViewProcessor

        self.logger = logger
        self.device = device
        self.models_dir = models_dir
        self.camera_config = camera_config
        self.texture_size = texture_size
        self.view_size = view_size

        class Config:
            pass

        self.config = Config()
        self.config.device = device
        self.config.multiview_cfg_path = str(official_repo / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
        self.config.custom_pipeline = "hunyuanpaintpbr"
        self.config.raster_mode = "cr"
        self.config.bake_mode = "back_sample"
        self.config.render_size = view_size
        self.config.texture_size = texture_size
        self.config.max_selected_view_num = len(camera_config.azims)
        self.config.resolution = view_size
        self.config.bake_exp = 4
        self.config.merge_method = "fast"
        self.config.ortho_scale = camera_config.ortho_scale
        self.config.candidate_camera_azims = list(camera_config.azims)
        self.config.candidate_camera_elevs = list(camera_config.elevs)
        self.config.candidate_view_weights = list(camera_config.weights)

        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
            ortho_scale=self.config.ortho_scale,
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        self.model = LocalMultiviewDiffusionNet(
            model_dir=models_dir / "hunyuan3d-paintpbr-v2-1",
            cfg_path=Path(self.config.multiview_cfg_path),
            dino_dir=models_dir / "dinov2-giant",
            device=device,
            logger=logger,
        )

    def load_mesh(self, mesh: trimesh.Trimesh) -> None:
        self.render.load_mesh(mesh=mesh)

    def generate_multiviews(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        steps: int,
        guidance_scale: float,
        seed: int,
        unwrap: bool = False,
    ) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image], List[Image.Image]]:
        from utils.uvwrap_utils import mesh_uv_wrap

        if unwrap:
            self.logger.info("UV unwrapping mesh inside paint stage.")
            mesh = mesh_uv_wrap(mesh)

        self.render.load_mesh(mesh=mesh)

        normal_maps = self.view_processor.render_normal_multiview(
            self.camera_config.elevs,
            self.camera_config.azims,
            use_abs_coor=True,
        )
        position_maps = self.view_processor.render_position_multiview(
            self.camera_config.elevs,
            self.camera_config.azims,
        )

        style_image = rgba_to_rgb_white(image).resize((self.view_size, self.view_size))
        multiviews_pbr = self.model(
            images=[style_image],
            conditions=normal_maps + position_maps,
            prompt="high quality",
            custom_view_size=self.view_size,
            resize_input=True,
            num_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        return multiviews_pbr["albedo"], multiviews_pbr["mr"], normal_maps, position_maps

    def bake_from_multiview(
        self,
        albedo_views: List[Image.Image],
        mr_views: List[Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        texture, mask = self.view_processor.bake_from_multiview(
            albedo_views,
            self.camera_config.elevs,
            self.camera_config.azims,
            self.camera_config.weights,
        )
        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            mr_views,
            self.camera_config.elevs,
            self.camera_config.azims,
            self.camera_config.weights,
        )
        return texture, mask, texture_mr, mask_mr

    def inpaint(
        self,
        albedo: torch.Tensor,
        albedo_mask: torch.Tensor,
        mr: torch.Tensor,
        mr_mask: torch.Tensor,
        vertex_inpaint: bool = True,
        method: str = "NS",
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask_np = mask_tensor_to_uint8(albedo_mask)
        texture_np = self.view_processor.texture_inpaint(albedo, mask_np, vertex_inpaint, method)

        mr_mask_np = mask_tensor_to_uint8(mr_mask)
        texture_mr_np = self.view_processor.texture_inpaint(mr, mr_mask_np, vertex_inpaint, method)

        return texture_np, texture_mr_np

    def set_texture_albedo(self, texture_np: np.ndarray) -> None:
        self.render.set_texture(texture_np, force_set=True)

    def set_texture_mr(self, texture_np: np.ndarray) -> None:
        self.render.set_texture_mr(texture_np)

    def save_mesh(self, output_obj_path: Path) -> Path:
        from convert_utils import create_glb_with_pbr_materials

        output_obj_path.parent.mkdir(parents=True, exist_ok=True)
        self.render.save_mesh(str(output_obj_path), downsample=False)
        output_glb_path = output_obj_path.with_suffix(".glb")
        textures = {
            "albedo": str(output_obj_path.with_suffix(".jpg")),
            "metallic": str(output_obj_path.with_name(output_obj_path.stem + "_metallic.jpg")),
            "roughness": str(output_obj_path.with_name(output_obj_path.stem + "_roughness.jpg")),
        }
        create_glb_with_pbr_materials(str(output_obj_path), textures, str(output_glb_path))
        return output_glb_path

    def clean_memory(self) -> None:
        cleanup_cuda()


def load_shape_pipeline(models_dir: Path, device: str):
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    ckpt_path = models_dir / "hunyuan3d-dit-v2-1" / "model.fp16.ckpt"
    config_path = models_dir / "hunyuan3d-dit-v2-1" / "config.yaml"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing shape checkpoint: {ckpt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing shape config: {config_path}")

    return Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=str(ckpt_path),
        config_path=str(config_path),
        device=device,
        dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
        use_safetensors=False,
    )


def generate_shape_latent(
    pipeline,
    image: Image.Image,
    steps: int,
    guidance_scale: float,
    seed: int,
):
    generator = torch.Generator(device=pipeline.device if hasattr(pipeline, "device") else "cpu").manual_seed(seed)
    try:
        return pipeline(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
        )
    except TypeError:
        return pipeline(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )


def load_shape_vae(models_dir: Path, device: str):
    from hy3dshape.models.autoencoders import ShapeVAE

    ckpt_path = models_dir / "hunyuan3d-vae-v2-1" / "model.fp16.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing VAE checkpoint: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    vae = ShapeVAE(**VAE_CONFIG)
    vae.load_state_dict(state, strict=False)
    vae.eval()
    vae = vae.to(dtype=torch.float16 if str(device).startswith("cuda") else torch.float32)
    vae = vae.to(device)
    return vae


def decode_latent_to_mesh(
    vae,
    latents: torch.Tensor,
    octree_resolution: int = 384,
    mc_level: float = 0.0,
    num_chunks: int = 8000,
    mc_algo: str = "mc",
    box_v: float = 1.01,
) -> trimesh.Trimesh:
    latents = vae.decode(latents)
    outputs = vae.latents2mesh(
        latents,
        output_type="trimesh",
        bounds=box_v,
        mc_level=mc_level,
        num_chunks=num_chunks,
        octree_resolution=octree_resolution,
        mc_algo=mc_algo,
        enable_pbar=True,
    )[0]
    outputs.mesh_f = outputs.mesh_f[:, ::-1]
    mesh = trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f, process=False)
    return mesh


def maybe_uv_wrap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    from utils.uvwrap_utils import mesh_uv_wrap
    return mesh_uv_wrap(mesh)


def build_camera_config() -> CameraConfig:
    return CameraConfig(
        azims=list(DEFAULT_CAMERA_AZIMS),
        elevs=list(DEFAULT_CAMERA_ELEVS),
        weights=list(DEFAULT_VIEW_WEIGHTS),
        ortho_scale=1.0,
    )


def choose_primary_image(args) -> Path:
    if args.image:
        return Path(args.image)
    if args.front:
        return Path(args.front)
    raise ValueError("Provide --image or --front at minimum.")


def collect_multiview_input_dict(args) -> Dict[str, Optional[Image.Image]]:
    out: Dict[str, Optional[Image.Image]] = {}
    for key in ("front", "left", "right", "back"):
        value = getattr(args, key)
        out[key] = load_image_rgba(Path(value)) if value else None
    return out


def save_generated_views(base_dir: Path, albedo: Sequence[Image.Image], mr: Sequence[Image.Image]) -> None:
    save_pil_list(base_dir / "multiview_albedo", albedo, "albedo")
    save_pil_list(base_dir / "multiview_mr", mr, "mr")


def configure_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("run_hunyuan3d")


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Hunyuan3D-2.1 generation and paint pipeline.")
    parser.add_argument("--image", type=str, default=None, help="Single reference image path.")
    parser.add_argument("--front", type=str, default=None, help="Front image path for experimental multi-view geometry input.")
    parser.add_argument("--left", type=str, default=None, help="Left image path for experimental multi-view geometry input.")
    parser.add_argument("--right", type=str, default=None, help="Right image path for experimental multi-view geometry input.")
    parser.add_argument("--back", type=str, default=None, help="Back image path for experimental multi-view geometry input.")
    parser.add_argument("--output", type=str, required=True, help="Final output path (.glb or .obj).")
    parser.add_argument("--steps", type=int, default=30, help="Number of diffusion steps for shape and paint.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale for shape; paint uses the same value.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--texture-size", type=int, default=1024, help="UV texture resolution.")
    parser.add_argument("--view-size", type=int, default=512, help="Rendered multiview conditioning resolution.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--no-remove-background", action="store_true", help="Skip background removal when alpha is not present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    seed_everything(args.seed)

    project_root = Path(__file__).resolve().parent
    official_repo = project_root / "third_party" / "Hunyuan3D-2.1"
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    add_repo_paths(official_repo, logger)

    from hy3dshape.rembg import BackgroundRemover

    primary_image_path = choose_primary_image(args)
    primary_rgba = load_image_rgba(primary_image_path)
    background_remover = BackgroundRemover()
    primary_rgba = maybe_remove_background(
        primary_rgba,
        remove_background=not args.no_remove_background,
        background_remover=background_remover,
    )
    save_pil_image(outputs_dir / "debug" / "input_rgba.png", primary_rgba)
    save_pil_image(outputs_dir / "debug" / "input_rgb_white.png", rgba_to_rgb_white(primary_rgba))

    mv_inputs = collect_multiview_input_dict(args)
    mv_count = sum(1 for img in mv_inputs.values() if img is not None)

    shape_pipe = load_shape_pipeline(models_dir=models_dir, device=args.device)

    if mv_count >= 2:
        logger.warning(
            "You supplied multiple geometry views, but the visualbruno multi-view geometry node is dormant and "
            "the referenced dit_config_2_1_mv.yaml is not present in the wrapper repo. "
            "This runnable script falls back to single-image geometry using --front/--image."
        )

    logger.info("Generating shape latent.")
    latents = generate_shape_latent(
        pipeline=shape_pipe,
        image=primary_rgba,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    cleanup_cuda()

    logger.info("Loading ShapeVAE and decoding latent to mesh.")
    vae = load_shape_vae(models_dir=models_dir, device=args.device)
    mesh = decode_latent_to_mesh(vae=vae, latents=latents)
    export_debug_mesh(outputs_dir / "debug" / "mesh_raw.glb", mesh)
    del vae
    cleanup_cuda()

    logger.info("Postprocessing mesh.")
    mesh = postprocess_mesh(mesh, logger=logger, remove_floaters=True, remove_degenerate_faces=True, reduce_faces=False)
    export_debug_mesh(outputs_dir / "debug" / "mesh_postprocessed.glb", mesh)

    logger.info("UV unwrapping mesh.")
    mesh = maybe_uv_wrap(mesh)
    export_debug_mesh(outputs_dir / "debug" / "mesh_uv.glb", mesh)

    camera_config = build_camera_config()
    logger.info(
        "Camera config: azims=%s elevs=%s weights=%s ortho_scale=%.3f",
        camera_config.azims,
        camera_config.elevs,
        camera_config.weights,
        camera_config.ortho_scale,
    )

    logger.info("Initializing split paint pipeline.")
    paint = StandalonePaintPipeline(
        models_dir=models_dir,
        official_repo=official_repo,
        camera_config=camera_config,
        texture_size=args.texture_size,
        view_size=args.view_size,
        device=args.device,
        logger=logger,
    )

    logger.info("Rendering normal and position maps, then running multiview paint.")
    albedo_views, mr_views, normal_maps, position_maps = paint.generate_multiviews(
        mesh=mesh,
        image=primary_rgba,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        unwrap=False,
    )

    save_pil_list(outputs_dir / "debug" / "normal_maps", normal_maps, "normal")
    save_pil_list(outputs_dir / "debug" / "position_maps", position_maps, "position")
    save_generated_views(outputs_dir / "debug", albedo_views, mr_views)

    logger.info("Baking multiview outputs into UV textures.")
    texture_albedo, mask_albedo, texture_mr, mask_mr = paint.bake_from_multiview(albedo_views, mr_views)
    save_texture_tensor(outputs_dir / "debug" / "baked_albedo.png", texture_albedo)
    save_mask_tensor(outputs_dir / "debug" / "baked_albedo_mask.png", mask_albedo)
    save_texture_tensor(outputs_dir / "debug" / "baked_mr.png", texture_mr)
    save_mask_tensor(outputs_dir / "debug" / "baked_mr_mask.png", mask_mr)

    logger.info("Running vertex-aware + OpenCV inpainting over UV textures.")
    texture_albedo_np, texture_mr_np = paint.inpaint(
        albedo=texture_albedo,
        albedo_mask=mask_albedo,
        mr=texture_mr,
        mr_mask=mask_mr,
        vertex_inpaint=True,
        method="NS",
    )
    save_pil_image(outputs_dir / "debug" / "inpainted_albedo.png", Image.fromarray(texture_albedo_np))
    save_pil_image(outputs_dir / "debug" / "inpainted_mr.png", Image.fromarray(texture_mr_np))

    logger.info("Applying textures and exporting mesh.")
    paint.load_mesh(mesh)
    paint.set_texture_albedo(texture_albedo_np)
    paint.set_texture_mr(texture_mr_np)

    requested_output = Path(args.output)
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    obj_path = requested_output.with_suffix(".obj") if requested_output.suffix.lower() != ".obj" else requested_output
    glb_path = paint.save_mesh(obj_path)
    paint.clean_memory()

    if requested_output.suffix.lower() == ".glb":
        shutil.copyfile(glb_path, requested_output)
    elif requested_output.suffix.lower() != ".obj":
        shutil.copyfile(glb_path, requested_output.with_suffix(".glb"))

    logger.info("OBJ saved at: %s", obj_path)
    logger.info("GLB saved at: %s", glb_path)
    logger.info("Final requested output: %s", requested_output)
    cleanup_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Validation, Debugging, and Failure Modes

The commands below are the fastest way to validate the environment before you attempt a full run.

### Core validation commands

```bash
# Conda + CUDA sanity
source ./conda/etc/profile.d/conda.sh
conda activate hunyuan3d-standalone

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

```bash
# Check the extensions import
python - <<'PY'
import custom_rasterizer_kernel
print("custom_rasterizer_kernel: OK")
PY
```

```bash
# Check mesh inpaint extension
python - <<'PY'
import sys
from pathlib import Path
repo = Path("third_party/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer").resolve()
sys.path.insert(0, str(repo))
import mesh_inpaint_processor
print("mesh_inpaint_processor: OK")
PY
```

```bash
# Check model files
find models -maxdepth 2 -type f | sort
```

```bash
# Run the pipeline
python run_hunyuan3d.py \
  --image inputs/front.png \
  --output outputs/model.glb \
  --steps 30 \
  --guidance-scale 5.0 \
  --seed 0 \
  --texture-size 1024 \
  --view-size 512 \
  --device cuda
```

The script saves all important intermediates under `outputs/debug/`: the input image after preprocessing, the raw mesh, the postprocessed mesh, the UV-wrapped mesh, normal maps, position maps, multiview albedo/MR outputs, baked textures, masks, and inpainted textures. That makes it much easier to isolate whether a bug starts in geometry, rendering, diffusion, UV bake, or export. citeturn22view0turn23view0turn25view6

### Common failure points

**CUDA mismatch**  
If `torch.version.cuda` does not match the CUDA runtime you intended to use, rebuild the environment instead of trying to patch it in place. The official tested stack is Torch 2.5.1 cu124; moving arbitrarily to newer wheel/runtime combinations is possible but increases extension-build risk. citeturn12view2turn38search0

**Missing `custom_rasterizer`**  
If normals or positions fail during rendering, or the texture stage crashes as soon as rasterization begins, the custom rasterizer is usually missing or compiled against the wrong CUDA toolchain. It is a required CUDA extension, not an optional speedup. citeturn42view1

**`libcudart` / extension load errors**  
These usually mean the torch wheel, system CUDA toolkit, and compiled extensions were built against different CUDA ABI expectations. Rebuild the extensions inside the exact environment where torch is installed. citeturn42view0turn42view1

**xformers / attention-mode confusion**  
The active wrapper path does not require xformers, and its shape single-file loader defaults to `attention_mode="sdpa"`. If xformers refuses to install or imports the wrong torch version, skip it first and verify the pipeline with pure SDPA. citeturn34view2turn43search0

**HF download failures**  
The paint split flow expects the local `hunyuan3d-paintpbr-v2-1` folder and the DINO folder to exist. If those downloads partially fail, paint initialization will die before rendering even starts. citeturn23view1turn17view0

**Bad alpha masks**  
If geometry is distorted or hollow, check `outputs/debug/input_rgba.png`. The shape image processor is built around a centered object with a modest border, so clipped silhouettes or giant background margins hurt downstream conditioning. citeturn28view0

**Black or empty textures**  
If baked textures are black, inspect:
- the normal and position renders
- the multiview albedo images
- the baked trust masks  
If the trust mask is nearly empty, the renderer or UV unwrap likely failed before diffusion quality even matters. citeturn23view0turn25view1

**UV unwrap failures**  
The wrapper UV step is a direct xatlas call. If that crashes or hangs, reduce mesh complexity first, then retry unwrap. Extremely pathological topology can also break xatlas outright. citeturn24view3

**`meshVerticeInpaint` build failures**  
If that extension is missing, OpenCV inpaint still exists, but the best mesh-aware result is gone. You will usually see worse seams and larger unresolved UV holes. citeturn25view6turn42view0

**GLB export problems**  
If the OBJ saves correctly but GLB is broken, verify that these files exist next to the OBJ:
- `your_mesh.jpg`
- `your_mesh_metallic.jpg`
- `your_mesh_roughness.jpg`  
The GLB conversion step assumes those names when building the PBR material. citeturn22view0turn39search2

### Open questions and limitations

The only major unresolved area is **true multiview geometry**. The wrapper clearly contains a commented node for it, but the required `dit_config_2_1_mv.yaml` is missing from the repo, so that path is not currently reproducible from the published code alone. Everything else in this tutorial is based on active code paths that exist today. citeturn10view1turn29view1turn28view1

The practical end command is:

```bash
python run_hunyuan3d.py --image inputs/front.png --output outputs/model.glb --steps 30 --texture-size 1024
```