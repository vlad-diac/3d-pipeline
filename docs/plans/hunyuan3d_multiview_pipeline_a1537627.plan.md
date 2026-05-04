---
name: Hunyuan3D Multiview Pipeline
overview: Implement the Hunyuan3D multiview pipeline following the consolidated plan, phased for incremental testing. Each phase produces testable output before proceeding.
todos:
  - id: phase1-setup
    content: "Phase 1: Create setup.sh, requirements.txt, download_models.py, and directory scaffold"
    status: completed
  - id: phase2-config-preprocess
    content: "Phase 2: Create config.py and preprocess.py"
    status: completed
  - id: phase3-mesh
    content: "Phase 3: Create mesh_generate.py and mesh_postprocess.py"
    status: completed
  - id: phase4-render-paint
    content: "Phase 4: Create render_multiview.py and paint_multiview.py"
    status: pending
  - id: phase5-bake-export-cli
    content: "Phase 5: Create bake_texture.py, inpaint_texture.py, export_glb.py, and main.py CLI"
    status: pending
isProject: false
---

# Hunyuan3D Multiview Pipeline Implementation

## Key Constraint: Dual-Platform

- **macOS (local dev)**: No CUDA, no GPU extension compilation. Setup script creates the conda env, installs CPU-compatible Python deps, clones repos, downloads models. Pipeline modules are importable but GPU-dependent stages (shape gen, paint, CUDA rasterizer) will only run on GPU.
- **RunPod (GPU testing)**: Full CUDA 12.4 stack, compiles CUDA extensions, runs end-to-end.

The setup script detects the platform and skips CUDA-only steps on macOS (with clear messages).

---

## Phase 1: Setup Script + Requirements + Directory Scaffold

Creates the full project skeleton and a `setup.sh` that bootstraps the environment on both macOS and RunPod.

### Files created:
- `setup.sh` -- main bootstrap (modeled on [scripts/diffusion/comfyui_install.sh](scripts/diffusion/comfyui_install.sh) `--continue` pattern)
- `requirements.txt` -- pinned Python deps from plan Section 3
- `scripts/download_models.py` -- HuggingFace model downloader
- `src/__init__.py` -- package init

### setup.sh steps (resumable via `--continue <step>`):
- **Step 0**: Pre-flight resource check (runs before any install)
  - Detect OS/arch (macOS arm64/x86_64, Linux x86_64)
  - Check if conda/miniconda is already installed and on PATH (or in `./conda/`)
  - Check if git is available
  - On Linux: check for `nvcc` / CUDA toolkit, `nvidia-smi`, `build-essential`/`gcc`
  - On macOS: note CUDA extensions will be skipped
  - Check available disk space (models need ~25 GB, repos ~2 GB)
  - Check if `third_party/` repos already cloned (skip re-clone)
  - Check if `models/` already has weights (skip re-download)
  - Print a summary table of what will be installed/skipped, then prompt `Continue? [Y/n]`
- **Step 1**: System deps (apt on Linux, skip on macOS)
- **Step 2**: Install Miniconda locally (into `./conda/`, skip if conda already found in Step 0)
- **Step 3**: Create conda env `hy3d-mv` with Python 3.10 (skip if env already exists)
- **Step 4**: Install PyTorch -- CUDA 12.4 on Linux, CPU on macOS
- **Step 5**: Install Python dependencies from `requirements.txt`
- **Step 6**: Clone `Hunyuan3D-2.1` and `Hunyuan3D-2` into `third_party/` (skip if present)
- **Step 7**: Build CUDA extensions (Linux/GPU only, skip on macOS with warning)
- **Step 8**: Download model weights via `download_models.py` (skip already-downloaded)
- **Step 9**: Verification checks

### requirements.txt scope:
All packages from plan Section 3 with exact pins. Platform-conditional packages (cupy, xformers) noted as comments for manual install on GPU.

### download_models.py:
Uses `huggingface_hub.snapshot_download` for each model in plan Section 2. Supports `--models-dir` and `--skip-existing` flags.

### Test gate (Phase 1):
```bash
# On macOS:
bash setup.sh
# Verify: conda env exists, imports work, third_party/ cloned, models/ populated

# On RunPod:
bash setup.sh
# Verify: above + CUDA extensions compiled + torch.cuda.is_available()
```

---

## Phase 2: Config + Preprocessing

- `src/config.py` -- `PipelineConfig` + `CameraConfig` dataclasses (plan Section 7)
- `src/preprocess.py` -- image loading, rembg, gray/white compose (plan Steps A+B)

### Test gate: load an image, remove background, compose over gray -- all CPU-testable on macOS.

---

## Phase 3: Mesh Generation + Postprocessing

- `src/mesh_generate.py` -- load shape pipeline, generate mesh (plan Steps C+D)
- `src/mesh_postprocess.py` -- floater removal, decimation, normalize (plan Step E)

### Test gate: RunPod only (requires GPU). Produce and save a raw + postprocessed `.glb`.

---

## Phase 4: UV Unwrap + Rendering + Paint

- `src/render_multiview.py` -- UV unwrap (xatlas), PaintPipeline init, normal/position map rendering (plan Steps F+G+H+I)
- `src/paint_multiview.py` -- delight, MultiviewDiffusionNet, upscale (plan Steps J+K+L)

### Test gate: RunPod -- produce normal maps, position maps, multiview albedo/MR images.

---

## Phase 5: Bake + Inpaint + Export + CLI

- `src/bake_texture.py` -- cosine-weighted UV back-projection (plan Step M)
- `src/inpaint_texture.py` -- vertex-aware + cv2 inpaint (plan Step N)
- `src/export_glb.py` -- PBR GLB export (plan Step O)
- `main.py` -- CLI entry point wiring all stages (plan Section 9)

### Test gate: RunPod -- full end-to-end: image in, textured `.glb` out.

---

## Implementation Notes

- The `--continue <step>` pattern from [scripts/diffusion/comfyui_install.sh](scripts/diffusion/comfyui_install.sh) is reused for `setup.sh` to allow resuming after failures.
- All `sys.path` manipulation is confined to `src/__init__.py` (adds `third_party/Hunyuan3D-2.1` and `third_party/Hunyuan3D-2` to path).
- `--save-intermediates` flag in `main.py` writes numbered stage directories per plan Section 10.
- Memory management: shape pipeline is deleted and `torch.cuda.empty_cache()` called before paint pipeline loads (plan Section 8).
