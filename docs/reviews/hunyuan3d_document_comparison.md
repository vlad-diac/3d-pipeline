# Hunyuan3D-2.1 Document Comparison Review

**Documents under review:**

| Label | File | Shorthand |
|-------|------|-----------|
| **Standalone Pipeline Tutorial** | `tutorials/hunyuan3d_2_1_standalone_pipeline.md` | **Tutorial** |
| **Research Document** | `docs/3d/hunyuan3d-python-1.md` | **Research** |

---

## Summary Table

| Section / Stage | Verdict | Synopsis |
|---|---|---|
| **A. Load Input Image** | DIVERGENCE | Research adds `load_image_rgba`, EXIF handling missing; Tutorial has EXIF but weaker RGBA handling |
| **B. Preprocess / Background** | DIVERGENCE | Research is more robust (alpha-extrema check, remover injection); Tutorial is simpler but fragile |
| **C. Load Shape Pipeline** | DIVERGENCE | Research uses `from_single_file` only with local ckpt; Tutorial offers both `from_single_file` and `from_pretrained` but passes wrong kwargs |
| **D. Generate Shape Latent** | DIVERGENCE | Research has `output_type="latent"` fallback via `try/except TypeError`; Tutorial uses `output_type="trimesh"` in runnable code (contradicts its own Stage D prose) |
| **E. Load ShapeVAE** | DIVERGENCE | Research correctly identifies VAE as **separate** checkpoint (`hunyuan3d-vae-v2-1/`); Tutorial wrongly says VAE is "embedded" in the shape checkpoint |
| **F. Decode Latent → Mesh** | MATCH | Both call `vae.decode()` → `vae.latents2mesh()` → flip face winding; identical parameters |
| **G. Postprocess Mesh** | DIVERGENCE | Research uses pure trimesh/pymeshlab postprocessing; Tutorial uses upstream `FloaterRemover` / `DegenerateFaceRemover` / `FaceReducer` classes |
| **H. UV Unwrap** | MATCH | Both use `xatlas.parametrize` identically; Research also offers `mesh_uv_wrap` from upstream utils |
| **I. Camera Config** | MATCH | Same defaults: azims, elevs, weights, `ortho_scale=1.0` |
| **J. Paint Pipeline Init** | DIVERGENCE | Research builds `StandalonePaintPipeline` with `ortho_scale` passed to `MeshRender`; Tutorial omits `ortho_scale` |
| **K. Render Normals/Positions** | MATCH | Identical calls to `render_normal_multiview` / `render_position_multiview` |
| **L. Multiview Paint Diffusion** | DIVERGENCE | Research implements `LocalMultiviewDiffusionNet` with DINO gating, scheduler swap, PBR mode split; Tutorial just calls `paint_pipeline.models["multiview_model"]` |
| **M. Bake Textures** | MATCH | Both use `bake_from_multiview` with same arguments |
| **N. Inpaint** | MATCH | Both use vertex-aware + OpenCV inpainting; Research passes method/vertex_inpaint explicitly |
| **O. Apply Textures** | MATCH | Both call `set_texture(..., force_set=True)` and `set_texture_mr(...)` |
| **P. Export GLB/OBJ** | DIVERGENCE | Research calls `create_glb_with_pbr_materials` correctly; Tutorial uses a fragile shutil-based fallback |
| **Environment Setup** | DIVERGENCE | Significant `requirements.txt` version differences; Research is more carefully pinned |
| **Multiview Geometry** | MATCH | Both correctly identify it as dormant; Research adds the runtime fallback + warning |
| **Camera/Texture Theory** | GAP | Tutorial has much more detailed theoretical explanation; Research is terse but code-accurate |

---

## Detailed Section-by-Section Analysis

---

### Stage A: Load Input Image

**Tutorial (lines 306–326):** Uses `ImageOps.exif_transpose()` for EXIF rotation, provides `image_to_tensor` and `tensor_to_pil` helpers. In the full runnable code (line 1399), loads with EXIF handling.

**Research (lines 536–538):** Uses `Image.open(path).convert("RGBA")` via `load_image_rgba`. No EXIF handling. Provides separate `choose_primary_image` (line 1036) and `collect_multiview_input_dict` (line 1044) functions.

| Criterion | Winner |
|---|---|
| EXIF handling | Tutorial |
| RGBA consistency | Research (always converts to RGBA) |
| Multi-view input support | Research (has `collect_multiview_input_dict`) |
| Error handling | Research (validates `getbbox() is None` for transparent images) |

**Verdict: DIVERGENCE** — Research is more production-ready due to alpha validation and multi-view input helpers. Tutorial has better EXIF handling. Neither is complete alone.

---

### Stage B: Preprocess / Background

**Tutorial (lines 343–355, runnable at 1394–1411):** Simple `if remove_bg and img.mode == 'RGB'` check. Creates a new `BackgroundRemover()` instance on every call inside the runnable code. No alpha-extrema gating.

**Research (lines 540–549):** Checks `image.getchannel("A").getextrema() == (255, 255)` before running the remover — this means it only removes background when the alpha channel is fully opaque (no existing transparency). Accepts `background_remover` as an injected parameter, avoiding redundant construction.

| Criterion | Winner |
|---|---|
| Alpha intelligence | Research — checks alpha extrema before running rembg |
| Dependency injection | Research — remover is passed in |
| Simplicity | Tutorial |

**Verdict: DIVERGENCE** — Research is better. The Tutorial's approach (`if img.mode == 'RGB'`) would skip background removal on RGBA images with fully opaque alpha, which is wrong. The Research's extrema check catches that case.

---

### Stage C: Load Shape Pipeline

**Tutorial (lines 373–407, runnable at 1428–1441):**
```python
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
    config_path=config_path,
    ckpt_path=ckpt_path,
    device=device,
    attention_mode=attention_mode
)
# OR
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    subfolder='hunyuan3d-dit-v2-1',
    device=device,
)
```

**Research (lines 937–953):**
```python
Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
    ckpt_path=str(ckpt_path),
    config_path=str(config_path),
    device=device,
    dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
    use_safetensors=False,
)
```

| Criterion | Finding |
|---|---|
| `from_pretrained` support | Tutorial offers it; Research does not |
| `dtype` parameter | Research passes `dtype=torch.float16`; Tutorial omits it (defaults to float32 on many setups) |
| `use_safetensors` | Research passes `use_safetensors=False` for `.ckpt` files; Tutorial omits |
| `attention_mode` parameter | Tutorial passes it; Research omits but notes wrapper defaults to `"sdpa"` in prose |
| Error handling on missing files | Research has explicit `FileNotFoundError` checks; Tutorial does not |

**Verdict: DIVERGENCE** — Research is more correct for local-checkpoint workflows (proper dtype, safetensors flag, file validation). Tutorial is more flexible by offering both loading paths. The Tutorial's `from_pretrained` code path is reasonable but may not actually pass `device` correctly depending on the pipeline's signature.

---

### Stage D: Generate Shape Latent

This is a **critical discrepancy**.

**Tutorial Stage D prose (lines 424–457):** Describes `output_type="latent"` to defer VAE decode.

**Tutorial full runnable code (lines 1449–1456):**
```python
meshes = pipeline(
    image=image,
    ...
    output_type="trimesh",  # ← CONTRADICTS the prose!
)
```

**Research (lines 956–978):**
```python
try:
    return pipeline(
        image=image,
        ...
        output_type="latent",
    )
except TypeError:
    return pipeline(
        image=image,
        ...
    )
```

| Criterion | Finding |
|---|---|
| Split flow (latent first, VAE later) | Research implements it correctly; Tutorial's runnable code does NOT |
| `output_type="latent"` fallback | Research has `try/except TypeError` for API compatibility; Tutorial doesn't |
| Consistency | Tutorial's prose says "latent" but runnable code says "trimesh" — internal contradiction |
| Generator device | Research uses `torch.Generator(device=...)` ; Tutorial uses `torch.manual_seed()` (device-agnostic, less precise) |

**Verdict: DIVERGENCE — Research is correct.** The Tutorial's runnable code at line 1455 uses `output_type="trimesh"`, which means it never actually does a split latent→VAE→mesh flow. The Tutorial then completely skips Stages E and F in the runnable code. The Research correctly implements the split and adds a `TypeError` fallback for API compatibility.

---

### Stage E: Load ShapeVAE

**This is the most significant factual disagreement between the two documents.**

**Tutorial (lines 463–519):** Says at line 515:
> "When using `from_pretrained`, the VAE is embedded in the single checkpoint alongside the DiT model."

The Tutorial's model table (line 214) lists `Hunyuan3D-VAE (embedded)` with description "Part of shape checkpoint". The Tutorial does NOT download the VAE separately in `download_models.py`.

**Research (lines 151–157, 218–222):** Correctly identifies the VAE as a **separate** checkpoint:
```
models/hunyuan3d-vae-v2-1/
    └── model.fp16.ckpt
```
The Research's download script (lines 218–222) explicitly downloads `hunyuan3d-vae-v2-1/*` as a separate HuggingFace subfolder. The prose (line 157) states the VAE is "about 656 MB" in its own folder.

**Research `load_shape_vae` (lines 981–994):**
```python
ckpt_path = models_dir / "hunyuan3d-vae-v2-1" / "model.fp16.ckpt"
state = torch.load(str(ckpt_path), ...)
vae = ShapeVAE(**VAE_CONFIG)
vae.load_state_dict(state, strict=False)
```

**Tutorial `load_shape_vae` (lines 473–512):** Provides a `model_path` parameter but notes that `from_pretrained` loads it automatically.

| Criterion | Finding |
|---|---|
| VAE as separate checkpoint | Research is **correct** — HF tree confirms `hunyuan3d-vae-v2-1/` is separate |
| VAE download | Research downloads it; Tutorial does **not** |
| VAE config values | Both use identical `VAE_CONFIG` dictionaries (cross-checked all 14 parameters — exact match) |
| `weights_only` parameter | Tutorial uses `weights_only=True`; Research uses `weights_only=False` — depends on checkpoint format |

**Verdict: DIVERGENCE — Research is correct.** The Tutorial's claim that the VAE is "embedded in the single checkpoint" is misleading. While `from_single_file` may split a combined checkpoint internally, the HuggingFace model tree has the VAE as a separate download under `hunyuan3d-vae-v2-1/`. The Tutorial's download script is therefore incomplete.

---

### Stage F: Decode Latent → Mesh

**Tutorial (lines 535–587):**
```python
vae.enable_flashvdm_decoder(enabled=enable_flash_vdm, mc_algo=mc_algo)
decoded = vae.decode(latents)
outputs = vae.latents2mesh(decoded, ..., bounds=box_v, ...)[0]
outputs.mesh_f = outputs.mesh_f[:, ::-1]
mesh = Trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f)
```

**Research (lines 997–1019):**
```python
latents = vae.decode(latents)
outputs = vae.latents2mesh(latents, ..., bounds=box_v, ...)[0]
outputs.mesh_f = outputs.mesh_f[:, ::-1]
mesh = trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f, process=False)
```

| Criterion | Finding |
|---|---|
| `enable_flashvdm_decoder` | Tutorial calls it; Research omits (safer default) |
| `process=False` in trimesh constructor | Research has it (prevents trimesh from modifying geometry); Tutorial omits |
| Face winding flip | Both do `mesh_f[:, ::-1]` |
| Cleanup | Tutorial does `torch.cuda.empty_cache()` + `gc.collect()`; Research defers to caller |

**Verdict: MATCH** — Functionally equivalent. Research's `process=False` is a better practice. Tutorial's `enable_flashvdm_decoder` is a nice-to-have.

---

### Stage G: Postprocess Mesh

**Tutorial (lines 604–632, runnable at 1472–1486):** Uses upstream Hunyuan classes:
```python
FloaterRemover()(mesh)
DegenerateFaceRemover()(mesh)
FaceReducer()(mesh, max_facenum=max_facenum)
```

**Research (lines 600–667):** Implements postprocessing with pure trimesh + pymeshlab:
```python
keep_largest_component(mesh)  # mesh.split → keep largest
remove_degenerate_and_duplicate_faces(mesh)  # trimesh built-in
decimate_mesh_with_pymeshlab(mesh, target_faces)  # pymeshlab quadric collapse
```

| Criterion | Winner |
|---|---|
| Dependency on upstream code | Tutorial depends on `hy3dshape.postprocessors`; Research is self-contained |
| Portability | Research — no upstream imports needed |
| Fidelity to wrapper behavior | Tutorial — directly uses same classes as ComfyUI wrapper |
| Default max_faces | Tutorial: `40000`; Research: `200000` |
| `reduce_faces` default | Tutorial: `True`; Research: `False` |

**Verdict: DIVERGENCE** — Tutorial is more faithful to the wrapper. Research is more portable. Notably, the Research defaults to `reduce_faces=False` and `max_faces=200000`, which is significantly different from the Tutorial's aggressive `40000`. The Tutorial's value matches the wrapper's default.

---

### Stage H: UV Unwrap

Both documents use `xatlas.parametrize(mesh.vertices, mesh.faces)` and rewrite `mesh.vertices`, `mesh.faces`, `mesh.visual.uv`. Research also imports `mesh_uv_wrap` from upstream utils for convenience.

**Verdict: MATCH**

---

### Stage I: Camera Config

Both documents use identical defaults:

| Parameter | Both |
|---|---|
| Azimuths | `[0, 90, 180, 270, 0, 180]` |
| Elevations | `[0, 0, 0, 0, 90, -90]` |
| Weights | `[1.0, 0.1, 0.5, 0.1, 0.05, 0.05]` |
| `ortho_scale` | `1.0` |

Research uses a `CameraConfig` dataclass (line 670); Tutorial returns a plain dict.

**Verdict: MATCH**

---

### Stage J: Paint Pipeline Init

**Tutorial (lines 736–775, runnable at 1518–1538):**
```python
config = Hunyuan3DPaintConfig(max_num_view=..., resolution=...)
config.texture_size = texture_size * 4
config.render_size = texture_size * 2
paint_pipeline = Hunyuan3DPaintPipeline(config)
```
In the stepwise variant (lines 1630–1636):
```python
render = MeshRender(
    default_resolution=config.render_size,
    texture_size=config.texture_size,
    bake_mode=config.bake_mode,
    raster_mode=config.raster_mode,
)
```
**`ortho_scale` is NOT passed to `MeshRender`.**

**Research (lines 817–823):**
```python
self.render = MeshRender(
    default_resolution=self.config.render_size,
    texture_size=self.config.texture_size,
    bake_mode=self.config.bake_mode,
    raster_mode=self.config.raster_mode,
    ortho_scale=self.config.ortho_scale,  # ← Present!
)
```

| Criterion | Finding |
|---|---|
| `ortho_scale` passed to `MeshRender` | **Research: YES, Tutorial: NO** |
| `bake_exp` config | Research sets `self.config.bake_exp = 4`; Tutorial omits |
| `merge_method` config | Research sets `"fast"`; Tutorial omits |
| `LocalMultiviewDiffusionNet` | Research builds it from scratch (lines 678–773); Tutorial relies on upstream `Hunyuan3DPaintPipeline` |

**Verdict: DIVERGENCE — Research is more correct.** Missing `ortho_scale` in the `MeshRender` constructor would cause the renderer to use its internal default, which may not match the camera config. The Research also explicitly sets `bake_exp` and `merge_method`, which affect texture quality.

---

### Stage K: Render Conditioning Maps

Both documents call:
```python
render_normal_multiview(elevs, azims, use_abs_coor=True)
render_position_multiview(elevs, azims)
```

**Verdict: MATCH**

---

### Stage L: Multiview Paint Diffusion

**Tutorial (lines 839–885):** Calls `paint_pipeline.models["multiview_model"](...)` as a black box. No scheduler swap, no DINO gating, no PBR mode detection.

**Research (lines 678–773):** Implements `LocalMultiviewDiffusionNet` from scratch:
- Loads `HunyuanPaintPipeline.from_pretrained` (line 688)
- Swaps scheduler to `EulerAncestralDiscreteScheduler` with `timestep_spacing="trailing"` (line 693)
- Checks `unet.use_dino` and conditionally loads `Dino_v2` (line 704)
- Splits output into `albedo` and `mr` based on PBR mode detection from config (line 768)
- Passes `sync_condition=None`, `dino_hidden_states` conditionally (lines 757–766)
- Uses `pipeline.enable_vae_slicing()` and `enable_vae_tiling()` for memory efficiency (lines 699–700)

| Criterion | Winner |
|---|---|
| Scheduler handling | Research — explicit Euler Ancestral swap matches wrapper |
| DINO feature conditioning | Research — gated on `unet.use_dino`; Tutorial ignores it |
| PBR mode detection | Research — checks config `custom_pipeline`; Tutorial assumes PBR |
| Memory optimization | Research — VAE slicing/tiling enabled |
| Reproducibility | Research — standalone class with no wrapper dependency |

**Verdict: DIVERGENCE — Research is significantly better.** The Tutorial's approach is essentially a wrapper around the upstream `Hunyuan3DPaintPipeline`, while the Research reverse-engineers the actual diffusion call with all necessary configuration.

---

### Stage M: Bake Textures

Both call `view_processor.bake_from_multiview(views, elevs, azims, weights)` identically.

**Verdict: MATCH**

---

### Stage N: Inpaint

Both use vertex-aware + OpenCV inpainting. Research explicitly passes `vertex_inpaint` and `method` parameters. Tutorial's stepwise variant calls `view_processor.texture_inpaint(texture, mask_np)` directly.

**Verdict: MATCH** — Functionally equivalent.

---

### Stage O: Apply Textures

Both call `render.set_texture(..., force_set=True)` and `render.set_texture_mr(...)`.

**Verdict: MATCH**

---

### Stage P: Export GLB/OBJ

**Tutorial (lines 1038–1078, runnable at 1739–1753):**
```python
render.save_mesh(output_obj, downsample=True)
from convert_utils import convert_obj_to_glb  # ← This function may not exist
convert_obj_to_glb(output_obj, glb_path)
```
The Tutorial's prose (line 1034) mentions `create_glb_with_pbr_materials` but the runnable code calls a different function `convert_obj_to_glb`. The simpler `generate_textures` function (line 1559) delegates to `paint_pipeline(mesh_path=..., save_glb=True)` and then does `shutil.move`.

**Research (lines 919–931):**
```python
from convert_utils import create_glb_with_pbr_materials
self.render.save_mesh(str(output_obj_path), downsample=False)
textures = {
    "albedo": str(output_obj_path.with_suffix(".jpg")),
    "metallic": str(output_obj_path.with_name(stem + "_metallic.jpg")),
    "roughness": str(output_obj_path.with_name(stem + "_roughness.jpg")),
}
create_glb_with_pbr_materials(str(output_obj_path), textures, str(output_glb_path))
```

| Criterion | Finding |
|---|---|
| GLB conversion function | Research uses `create_glb_with_pbr_materials` (correct per upstream); Tutorial uses `convert_obj_to_glb` (may not exist) |
| Texture file paths for GLB | Research explicitly constructs `_metallic.jpg`, `_roughness.jpg` names; Tutorial does not |
| `downsample` parameter | Research: `False`; Tutorial: `True` |
| Robustness | Research — predictable file layout; Tutorial — fragile `shutil.move` chain |

**Verdict: DIVERGENCE — Research is correct.** The Tutorial references `convert_obj_to_glb` which does not appear in the upstream `convert_utils.py` — the real function is `create_glb_with_pbr_materials`. The Research's explicit texture-path construction ensures the GLB builder finds the right files.

---

## Cross-Cutting Technical Verification

### 1. Split flow (latent before baking)

**Question:** Does the standalone pipeline correctly handle the split flow where the wrapper returns multiview images BEFORE baking?

| Document | Finding |
|---|---|
| Tutorial | **Partially.** The stepwise variant (lines 1592–1759) implements the split, but the default `generate_textures` (lines 1511–1586) calls `paint_pipeline(mesh_path=..., save_glb=True)` monolithically. |
| Research | **Yes.** The `StandalonePaintPipeline` class exposes `generate_multiviews`, `bake_from_multiview`, `inpaint`, and `save_mesh` as separate stages. The `main()` function calls them in sequence (lines 1167–1207). |

**Winner: Research**

### 2. VAE as separate checkpoint

**Question:** Does the standalone pipeline correctly identify that the VAE is a SEPARATE checkpoint from the DiT?

| Document | Finding |
|---|---|
| Tutorial | **No.** Line 214: "Hunyuan3D-VAE (embedded) — Part of shape checkpoint". Line 515: "the VAE is embedded in the single checkpoint". Download script omits VAE. |
| Research | **Yes.** Lines 153, 218–222: Explicitly downloads `hunyuan3d-vae-v2-1/*` separately. Prose at line 157: "The VAE lives under a separate `hunyuan3d-vae-v2-1/` folder, where `model.fp16.ckpt` is about 656 MB." |

**Winner: Research** — This is factually verified against the HuggingFace model tree.

### 3. `from_single_file` vs `from_pretrained` loading paths

| Document | Finding |
|---|---|
| Tutorial | Offers both paths but doesn't verify which kwargs each accepts. The `from_pretrained` call passes `device=` which may not be a valid parameter for all versions. |
| Research | Uses only `from_single_file` with `dtype`, `use_safetensors` kwargs. Cleaner and tested against the actual checkpoint format. |

**Winner: Research** for correctness; Tutorial for breadth of options.

### 4. `output_type="latent"` vs `output_type="trimesh"`

| Document | Finding |
|---|---|
| Tutorial | **Inconsistent.** Prose in Stage D says `"latent"`. Runnable code at line 1455 says `"trimesh"`. The runnable code therefore skips Stages E and F entirely. |
| Research | **Consistent.** Uses `output_type="latent"` with `try/except TypeError` fallback (lines 965–978). Then separately loads VAE and decodes (lines 1131–1133). |

**Winner: Research** — The Tutorial contradicts itself.

### 5. `requirements.txt` consistency

| Package | Tutorial | Research | Note |
|---|---|---|---|
| `diffusers` | `0.30.0` | `0.31.0` | Research is newer |
| `huggingface-hub` | `0.30.2` | `0.26.3` | Tutorial is newer |
| `cupy-cuda12x` | `13.4.1` | `13.3.0` | Minor difference |
| `pytorch-lightning` | `1.9.5` | `2.4.0` | **Major** — Tutorial uses legacy v1; Research uses v2 |
| `rembg` | `2.0.65` | `2.0.59` | Minor |
| `safetensors` | `0.4.4` | `0.4.5` | Minor |
| `trimesh` | `4.4.7` | `4.4.9` | Minor |
| `pymeshlab` | `2022.2.post3` | `2023.12.post2` | Research is newer |
| `pygltflib` | `1.16.3` | `1.16.2` | Minor |
| `pybind11` | `2.13.4` | `2.13.6` | Minor |
| `basicsr` | absent | `1.4.2` | Research includes it (needed for RealESRGAN) |
| `realesrgan` | absent | `0.3.0` | Research includes it |
| `onnxruntime` | absent | `1.19.2` | Research includes it (needed for rembg) |
| `timm` | unpinned | `1.0.11` | Research pins it |
| `pillow` | unpinned | `10.4.0` | Research pins it |
| `spandrel` | present | absent | Tutorial has it, Research doesn't |
| `numpy` | `1.24.4` | absent (inherits) | Tutorial pins, Research inherits from torch |
| `scipy` | `1.14.1` | absent | Tutorial includes, Research doesn't |
| `imageio` | `2.36.0` | absent | Tutorial includes, Research doesn't |
| `ninja` | `1.11.1.1` | absent | Tutorial includes, Research relies on system |
| `tqdm` | `4.66.5` | absent | Tutorial includes, Research inherits |
| `configargparse` | `1.7` | `1.7` | Match |
| `open3d` | `0.18.0` | `0.18.0` | Match |

**Verdict: DIVERGENCE** — Research's `requirements.txt` is more carefully curated for the actual paint pipeline (includes `basicsr`, `realesrgan`, `onnxruntime`). Tutorial includes some packages not needed (`scipy`, `imageio`, `spandrel`) but misses critical paint-stack dependencies.

### 6. `LocalMultiviewDiffusionNet` architecture

| Document | Finding |
|---|---|
| Tutorial | Does **not** reproduce this class. Relies on upstream `Hunyuan3DPaintPipeline` or `multiviewDiffusionNet` from utils. |
| Research | **Fully implements** `LocalMultiviewDiffusionNet` (lines 678–773) with scheduler swap, DINO gating, PBR mode detection, `seed_everything`, and control-image packing. |

**Winner: Research** — This is the most architecturally significant class in the paint stage, and only the Research document implements it.

### 7. `ortho_scale` passed to `MeshRender`

| Document | Finding |
|---|---|
| Tutorial | **Not passed.** Line 1630–1635 constructs `MeshRender` without `ortho_scale`. |
| Research | **Passed.** Line 822: `ortho_scale=self.config.ortho_scale`. |

**Winner: Research**

### 8. `create_glb_with_pbr_materials` for GLB export

| Document | Finding |
|---|---|
| Tutorial | Mentions it in prose (line 1034) but the runnable code uses `convert_obj_to_glb` (line 1749), which may not exist. The simpler flow delegates to `paint_pipeline(...)` which handles it internally. |
| Research | Calls it directly (line 930) with explicit texture file paths. |

**Winner: Research** — Uses the correct function name from `convert_utils.py`.

### 9. `output_type="latent"` support handling

| Document | Finding |
|---|---|
| Tutorial | Discusses it in prose but doesn't handle the case where the pipeline may not support it. Runnable code avoids the issue by using `output_type="trimesh"`. |
| Research | Wraps the call in `try/except TypeError` (lines 972–978) to fall back gracefully if the kwarg isn't supported. |

**Winner: Research** — Defensive coding for API compatibility.

---

## Environment Setup Comparison

### `setup.sh`

| Feature | Tutorial | Research |
|---|---|---|
| Conda installer | Miniconda (anaconda.com) | Miniforge (conda-forge) |
| System packages | Not installed | Installs `build-essential`, `libgl1`, etc. |
| Env name | `hy3d21` | `hunyuan3d-standalone` |
| xformers handling | Required install | Optional with `|| true` fallback |
| Model download | Manual (`download_models.py` run separately) | Integrated into `setup.sh` |
| Git LFS | Not mentioned | Installs `git-lfs` |
| Sudo handling | Not handled | Checks for `sudo` availability |
| Repo directory | `repos/` | `third_party/` |

**Verdict: DIVERGENCE** — Research setup.sh is more robust (handles rootless containers, installs system deps, makes xformers optional, integrates model download). Tutorial is simpler but assumes more about the host environment.

### CUDA Notes

Both mention the same tested stack (Python 3.10, PyTorch 2.5.1+cu124). Tutorial provides a more detailed compatibility table with driver versions and arch lists. Research adds context about the wrapper's different tested stack (Windows 11, Python 3.12, Torch 2.6.0+cu126).

---

## Multiview Geometry Variant

Both documents correctly identify:
- The variant is present as dormant/commented code in `nodes.py`
- `dit_config_2_1_mv.yaml` is missing from the repo
- The input contract expects a `{front, left, back, right}` dictionary
- It is not production-ready

**Key difference:** The Research's runnable code (lines 1114–1119) includes an actual **runtime warning** when the user supplies multiple view images, explaining the fallback to single-image geometry. The Tutorial only describes this in documentation.

**Verdict: MATCH** — Both are thorough. Research adds runtime safety.

---

## Camera + Texture Explanation

**Tutorial (lines 1198–1267):** Provides extensive theoretical explanation:
- Why six cameras are used
- How normal maps and position maps condition the paint model
- What view weights do (with weight values explained)
- Why UV wrapping must happen before baking (5-step algorithm)
- How cosine-weighted back-projection works (formula included)
- Why texture masks are trust masks, not segmentation masks

**Research (lines 417–427):** Much more concise. Covers the same points but in ~10 lines of prose rather than ~70 lines. Focuses on practical code-level observations:
- Cosine visibility raised to `bake_exp`
- Trust map threshold `> 1e-8`
- Two-step inpaint rationale

**Verdict: GAP** — The Tutorial has significantly more theoretical depth. The Research compensates with code-level specifics (like `bake_exp=4` and the trust threshold) that the Tutorial omits.

---

## Code Quality Comparison

### Error Handling

| Feature | Tutorial | Research |
|---|---|---|
| File existence checks | Minimal | Multiple `FileNotFoundError` raises |
| Alpha validation | None | `getbbox() is None` check |
| Import error handling | None | Deferred imports with meaningful messages |
| GPU cleanup | `gc.collect()` + `torch.cuda.empty_cache()` | Same, plus `cleanup_cuda()` as reusable function |
| CUDA availability check | Yes (in `main()`) | Implicit (via `device` parameter) |

### Structure

| Feature | Tutorial | Research |
|---|---|---|
| Typing | Partial (`List`, `Optional`) | More comprehensive (Tuple, Dict, Sequence, dataclass) |
| Data classes | None | `CameraConfig` dataclass |
| Logging | Built-in `logging` | Same, with logger injection |
| Intermediate saving | `--save-intermediates` flag | Always saves to `outputs/debug/` |
| Code organization | Functions + one `main()` | Functions + classes (`LocalMultiviewDiffusionNet`, `StandalonePaintPipeline`) |
| Lines of code | ~550 (runnable section) | ~780 (runnable section) |

### GPU Memory Management

| Feature | Tutorial | Research |
|---|---|---|
| Model offloading | `--low-vram` flag (declared but not fully implemented) | `del pipeline; cleanup_cuda()` after shape stage |
| VAE slicing/tiling | Not used | `enable_vae_slicing()`, `enable_vae_tiling()` on paint model |
| Sequential cleanup | `del pipeline; cleanup_gpu()` after shape | Same pattern, more consistent |

**Overall code quality winner: Research** — Better typing, error handling, memory management, and architectural separation via classes.

---

## Final Verdicts

### What the Tutorial does better:
1. **Theoretical explanations** — Camera, texture, and baking theory is significantly more detailed.
2. **CLI flexibility** — More command-line options (`--low-vram`, `--attention-mode`, `--save-intermediates` as a flag, separate `--paint-steps` / `--paint-guidance`).
3. **EXIF image handling** — `ImageOps.exif_transpose()`.
4. **Validation/debugging section** — More test commands and troubleshooting scenarios.
5. **`from_pretrained` option** — Allows HuggingFace-hosted model loading without local downloads.

### What the Research does better:
1. **VAE checkpoint handling** — Correctly identifies it as separate, downloads it, loads it independently.
2. **Split flow implementation** — Actually implements the latent→VAE→mesh split in the runnable code.
3. **`LocalMultiviewDiffusionNet`** — Fully reimplements the paint diffusion model loading with DINO gating and scheduler swap.
4. **`ortho_scale`** — Passed to `MeshRender` constructor.
5. **`create_glb_with_pbr_materials`** — Uses the correct function name with explicit texture paths.
6. **`output_type="latent"` fallback** — `try/except TypeError` for API compatibility.
7. **Error handling and file validation** — Multiple `FileNotFoundError` checks.
8. **Requirements.txt** — Includes critical paint-stack dependencies (`basicsr`, `realesrgan`, `onnxruntime`).
9. **Setup script** — Handles system packages, rootless containers, optional xformers.
10. **Memory optimization** — VAE slicing/tiling on paint pipeline.
11. **Runtime multiview fallback** — Warns and falls back when multi-view inputs are provided but the path is dormant.
12. **Internal consistency** — No contradictions between prose and runnable code.

### Critical bugs in the Tutorial:
1. **Line 1455:** `output_type="trimesh"` contradicts Stage D prose about `output_type="latent"`.
2. **Line 214:** "Hunyuan3D-VAE (embedded)" is factually incorrect.
3. **Lines 232–246:** Download script does not download the VAE separately.
4. **Line 1749:** `convert_obj_to_glb` — this function likely doesn't exist in `convert_utils.py`.
5. **Lines 1630–1635:** `MeshRender` constructor missing `ortho_scale` parameter.

### Critical bugs in the Research:
1. **No EXIF handling** — Images with EXIF rotation will appear wrong.
2. **`--steps` shared** — Same `--steps` value used for both shape and paint diffusion, which should typically differ (50 for shape, 10 for paint).
