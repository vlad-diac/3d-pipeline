# Addendum: Research Document 2 Analysis

**Appends to:** `reviews/hunyuan3d_document_comparison.md`

**New document under review:**

| Label | File | Shorthand |
|-------|------|-----------|
| **Research Document 2** | `docs/3d/hunyuan3d-python-2.md` | **Research 2** |

**Previously reviewed:**

| Label | File | Shorthand |
|-------|------|-----------|
| **Standalone Pipeline Tutorial** | `tutorials/hunyuan3d_2_1_standalone_pipeline.md` | **Tutorial** |
| **Research Document 1** | `docs/3d/hunyuan3d-python-1.md` | **Research 1** |

---

## Executive Summary

Research 2 is architecturally and conceptually different from both the Tutorial and Research 1. Where the first two documents build a **single-image geometry + multiview texture** pipeline around the `Hunyuan3D-2.1` repo, Research 2 builds a **four-view multiview geometry + multiview texture** pipeline that actively uses the `Hunyuan3D-2mv` (v2.0 multiview) shape model. It also introduces a **Delight / Light_Shadow_Remover** preprocessing step, a **gray-background composite** instead of white, a **different `Hunyuan3DPaintConfig` constructor signature**, and references `xatlas==0.0.10` instead of `0.0.9`. These are all novel compared to both previously reviewed documents.

---

## 1. NEW Information in Research 2

### 1.1 Delight / Light_Shadow_Remover Step

**Status: PRESENT in Research 2, ABSENT from Tutorial and Research 1.**

Research 2 introduces a dedicated light/shadow removal step before multiview paint diffusion:

```python
from hy3dgen.texgen.utils.dehighlight_utils import Light_Shadow_Remover

def load_delight_model(cfg):
    delight_cfg = SimpleNamespace(
        device=cfg.device,
        light_remover_ckpt_path=str(cfg.delight_model_dir),
    )
    return Light_Shadow_Remover(delight_cfg)

def delight_reference(front_rgba, cfg):
    reference = compose_over_gray(front_rgba, gray=127)
    delight = load_delight_model(cfg)
    if delight is None:
        return reference
    return delight(reference)
```

Key details:
- The import path is `hy3dgen.texgen.utils.dehighlight_utils`, which comes from the **Hunyuan3D-2.0** codebase (`Hunyuan3D-2`), NOT the 2.1 codebase. This is a deliberate cross-version dependency.
- The delight model loads a Diffusers pipeline from `light_remover_ckpt_path`, requiring a separate model download.
- This step processes the front reference image to reduce directional lighting bias *before* it enters the multiview paint diffusion. Research 2 explains: "Its job is to reduce directional lighting bias so the paint model is not forced to bake in strong highlights and shadows."
- The delight step is treated as optional — if the model directory doesn't exist, the pipeline proceeds without it.

Neither the Tutorial nor Research 1 mention this step. The Tutorial composites onto white and sends the result directly to the paint model. Research 1 does the same via `rgba_to_rgb_white`.

**Impact: This is a genuine pipeline improvement.** For production use with real-world photographs (as opposed to synthetic renders), delighting would reduce baked-in lighting artifacts in the final texture.

---

### 1.2 Gray Background Composite Instead of White

**Status: Research 2 uses `compose_over_gray(gray=127)`. Tutorial and Research 1 use white.**

| Document | Background for paint reference |
|---|---|
| Tutorial | `Image.new("RGB", ..., (255, 255, 255))` — white |
| Research 1 | `rgba_to_rgb_white()` — white |
| Research 2 | `compose_over_gray(front_rgba, gray=127)` — 50% gray |

Research 2 composites the RGBA reference image over a **neutral gray** background before feeding it to the delight step and then the multiview diffusion model. The Tutorial and Research 1 both composite over **white**.

This is not just a cosmetic difference: a neutral gray background gives the paint diffusion model a more balanced starting point and avoids biasing the network toward bright/washed-out border regions. It also aligns with the delight step's goal of removing directional lighting assumptions.

---

### 1.3 Multiview Shape Model (`hunyuan3d-dit-v2-mv`)

**Status: Research 2 uses a DIFFERENT shape model from the other documents.**

| Document | Shape model subfolder | Shape model approach |
|---|---|---|
| Tutorial | `hunyuan3d-dit-v2-1` | Single-image |
| Research 1 | `hunyuan3d-dit-v2-1` | Single-image |
| Research 2 | `hunyuan3d-dit-v2-mv` | **Four-view multiview** |

Research 2's `parse_args()` defaults `--shape-subfolder` to `"hunyuan3d-dit-v2-mv"`. Its `main()` function requires `--front`, `--left`, `--right`, `--back` as **mandatory** arguments — four-view input is the primary path, not an experimental one.

Research 2 explicitly references the `tencent/Hunyuan3D-2mv` HuggingFace repo and mentions fast/turbo variants:
> "Use `hunyuan3d-dit-v2-mv-fast` or `hunyuan3d-dit-v2-mv-turbo` instead of the standard 2mv checkpoint when geometry speed matters more than fidelity. Those variants are published directly in the 2mv repo."

This fundamentally contradicts the existing review's verdict that multiview geometry is "dormant." See the Corrections section below.

---

### 1.4 Different `Hunyuan3DPaintConfig` Constructor Signature

**Status: Research 2 uses a DIFFERENT constructor calling convention.**

| Document | Constructor call |
|---|---|
| Tutorial | `Hunyuan3DPaintConfig(max_num_view=..., resolution=...)` |
| Research 1 | `Hunyuan3DPaintConfig(max_num_view=..., resolution=...)` |
| Research 2 | `Hunyuan3DPaintConfig(resolution=..., camera_azims=..., camera_elevs=..., view_weights=..., ortho_scale=..., texture_size=...)` |

Research 2 passes camera arrays, ortho_scale, and texture_size **directly** to the constructor rather than monkey-patching them onto the config object after construction. The Tutorial and Research 1 both construct a minimal config and then mutate its attributes:

```python
# Tutorial / Research 1 pattern:
config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
config.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
config.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
config.candidate_view_weights = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]
```

```python
# Research 2 pattern:
conf = Hunyuan3DPaintConfig(
    resolution=cfg.view_size,
    camera_azims=cfg.camera_azimuths,
    camera_elevs=cfg.camera_elevations,
    view_weights=cfg.camera_weights,
    ortho_scale=cfg.ortho_scale,
    texture_size=cfg.texture_size,
)
```

If Research 2's constructor signature is correct, it implies either:
1. The upstream `Hunyuan3DPaintConfig` has been updated to accept these parameters directly, OR
2. Research 2 is referencing a different version of the paint config class.

**Note:** Research 2 does NOT include `max_num_view` in its constructor call, while the Tutorial and Research 1 both use it as a positional-style argument.

---

### 1.5 `xatlas==0.0.10` Instead of `0.0.9`

| Document | xatlas version |
|---|---|
| Tutorial | `0.0.9` |
| Research 1 | `0.0.9` |
| Research 2 | `0.0.10` |

Research 2 explicitly states: "The official 2.1 requirements pin `xatlas==0.0.10`." If this is true, both the Tutorial and Research 1 are using the wrong version.

---

### 1.6 Multiview Diffusion Parameters

Research 2 passes `num_steps`, `guidance_scale`, and `seed` to the multiview diffusion call:

```python
result = pipe.model(
    [reference_rgb],
    normal_maps + position_maps,
    prompt="high quality",
    custom_view_size=cfg.view_size,
    resize_input=True,
    num_steps=cfg.paint_steps,
    guidance_scale=cfg.paint_guidance_scale,
    seed=cfg.seed,
)
```

| Document | Passes `num_steps` | Passes `guidance_scale` | Passes `seed` |
|---|---|---|---|
| Tutorial | NO | NO | NO |
| Research 1 | YES | YES | YES |
| Research 2 | YES | YES | YES |

The existing review already noted Research 1's advantage here. Research 2 confirms this is the correct calling convention.

---

### 1.7 Mesh Normalization Step

Research 2 includes a `normalize_mesh` function not present in either other document:

```python
def normalize_mesh(mesh):
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    scale = float(np.max(bounds[1] - bounds[0]))
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale)
    return mesh
```

This is called at the end of `postprocess_mesh()`. Neither the Tutorial nor Research 1 normalize the mesh after postprocessing. This step ensures the mesh is centered at the origin and fits within a unit bounding box, which can be important for consistent renderer behavior.

---

### 1.8 Additional Postprocessing Calls

Research 2's `postprocess_mesh` includes two calls not present in the other documents:

- `mesh.remove_infinite_values()` — removes vertices with inf/NaN coordinates
- `mesh.merge_vertices()` — merges coincident vertices

It also uses `mesh.simplify_quadric_decimation(target_faces)` (trimesh's built-in) instead of pymeshlab or upstream `FaceReducer`.

---

### 1.9 Modular File Architecture

Research 2 splits the pipeline across multiple files under a `src/` package:

```
src/
├── config.py              # All workflow/path parameters
├── preprocess.py          # View normalization, compose_over_gray
├── mesh_generate.py       # Shape pipeline, postprocessing
├── render_multiview.py    # UV unwrap, normal/position rendering
├── paint_multiview.py     # Delight, multiview diffusion
├── bake_texture.py        # UV bake
├── inpaint_texture.py     # Mesh-aware + cv2 inpainting
├── export_glb.py          # Texture assignment + GLB conversion
└── main.py                # Orchestrator
```

The Tutorial and Research 1 are both single-file architectures. This modular design makes it easier to test, debug, and replace individual stages.

---

### 1.10 Performance Variants and Optimization Knobs

Research 2 documents performance optimization options not mentioned elsewhere:

- **Fast/turbo 2mv checkpoints:** `hunyuan3d-dit-v2-mv-fast`, `hunyuan3d-dit-v2-mv-turbo` from the `tencent/Hunyuan3D-2mv` repo
- **Reduce `render_size`** from 1024 to 768
- **Skip Delight** if inputs are studio-lit
- **Use the 2.0 fast multiview texture path** via `examples/fast_texture_gen_multiview.py`
- **Model CPU offload, VAE slicing, VAE tiling** mentioned as paint pipeline optimizations

---

### 1.11 `torch.inference_mode()` Usage

Research 2 wraps all inference stages in `torch.inference_mode()` context managers rather than `@torch.no_grad()` decorators. `inference_mode` is strictly more efficient than `no_grad` (it disables autograd entirely, not just gradient computation), and is the recommended approach for pure inference in PyTorch >= 1.9.

---

### 1.12 Numbered Stage Directories for Intermediates

Research 2 saves intermediates in numbered directories:

```
01_preprocessed/
02_mesh/
03_normal_maps/
04_position_maps/
05_reference/
06_mv_albedo/
07_mv_mr/
08_bake/
09_refine/
```

This is more organized than the Tutorial's flat `intermediates/` directory or Research 1's `outputs/debug/` directory.

---

## 2. Corrections to the Tutorial and Research 1

### 2.1 Multiview Geometry Is NOT Fully Dormant

The existing review states both documents "correctly identify [multiview geometry] as dormant." Research 2 challenges this — it successfully references the `Hunyuan3D-2mv` repo and the public `from_pretrained` API with multiview dictionary input. Research 2 cites the official Gradio app as evidence that the public pipeline signature accepts `dict` / `List[dict]` image input.

However, this is nuanced:
- The **ComfyUI wrapper's** multiview geometry node IS dormant (confirmed by all three documents).
- The **official Hunyuan3D-2** (v2.0) repo's multiview shape pipeline IS functional via `from_pretrained` with the 2mv checkpoint.
- The **Hunyuan3D-2.1** repo's multiview path remains unclear.

Research 2 solves this by using the 2.0 multiview shape model (`hunyuan3d-dit-v2-mv`) instead of trying to use the dormant 2.1 multiview path.

### 2.2 Delight Step Should Be Documented

Neither the Tutorial nor Research 1 mentions the delight/light-shadow-remover step. Research 2 demonstrates it is part of the upstream 2.0 texture pipeline and can improve results on real-world photographs. Both previous documents should note this as an available preprocessing improvement.

### 2.3 `xatlas` Version Should Be `0.0.10`

If Research 2 is correct that the official 2.1 requirements pin `xatlas==0.0.10`, both the Tutorial's and Research 1's `requirements.txt` files should be updated.

### 2.4 Paint Reference Should Consider Gray Background

Research 2's use of a gray background composite (instead of white) for the paint reference image is a potentially significant quality improvement that both other documents miss.

---

## 3. Contradictions Between Research 2 and Other Documents

### 3.1 Primary Shape Pipeline Architecture

| Aspect | Tutorial + Research 1 | Research 2 |
|---|---|---|
| Shape input | Single image | Four views (front/left/right/back) |
| Shape model | `hunyuan3d-dit-v2-1` | `hunyuan3d-dit-v2-mv` |
| Shape repo | Hunyuan3D-2.1 | Hunyuan3D-2 (v2.0 2mv) |
| Multiview status | Dormant / experimental | **Primary production path** |

This is the largest architectural contradiction. Research 2 builds around a fundamentally different geometry pipeline. It uses the v2.0 multiview shape model rather than the v2.1 single-image model.

### 3.2 Background Composite Color

| Document | Composite background |
|---|---|
| Tutorial | White `(255, 255, 255)` |
| Research 1 | White `(255, 255, 255)` |
| Research 2 | Gray `(127, 127, 127)` |

### 3.3 Paint Config Constructor

| Document | Constructor pattern |
|---|---|
| Tutorial | `Hunyuan3DPaintConfig(max_num_view=N, resolution=R)` + attribute mutation |
| Research 1 | Same as Tutorial |
| Research 2 | `Hunyuan3DPaintConfig(resolution=R, camera_azims=..., camera_elevs=..., view_weights=..., ortho_scale=..., texture_size=...)` |

### 3.4 Latent API Stance

| Document | Position on latent API |
|---|---|
| Tutorial | Uses `output_type="trimesh"` in runnable code (contradicts prose) |
| Research 1 | Uses `output_type="latent"` with `try/except TypeError` fallback |
| Research 2 | **Recommends against using latent API** for production; explicitly calls it "internal / unstable API territory." Recommends `from_pretrained` which returns a mesh directly. |

This is a meaningful philosophical difference. Research 1 implements the split latent→VAE→mesh flow that the existing review praises. Research 2 argues that flow is fragile and recommends the public `from_pretrained` → mesh API instead.

### 3.5 Shape Pipeline Loading

| Document | Loading method |
|---|---|
| Tutorial | `from_single_file` or `from_pretrained` |
| Research 1 | `from_single_file` only |
| Research 2 | `from_pretrained` only (public API) |

Research 2 explicitly discourages `from_single_file` for production and recommends `from_pretrained` with the HuggingFace-hosted checkpoint.

### 3.6 Mesh Decimation Implementation

| Document | Decimation approach | Default target |
|---|---|---|
| Tutorial | Upstream `FaceReducer` | 40,000 faces |
| Research 1 | pymeshlab `quadric_edge_collapse` | 200,000 faces |
| Research 2 | trimesh `simplify_quadric_decimation` | 200,000 faces |

Research 2 agrees with Research 1 on the 200k face target but uses a different implementation (trimesh built-in vs pymeshlab).

---

## 4. Latest API Information

### 4.1 `from_pretrained` for Shape Model

Research 2 is the only document that treats `from_pretrained` as the recommended production API:

> "For the production script, use the public pipeline above. That is the path with the most stable API surface."

It references the official Gradio app (`gradio_app.py`) as evidence that `from_pretrained` with dictionary input is the intended public interface.

### 4.2 HuggingFace Examples Referenced

Research 2 cites several upstream examples not referenced by the other documents:

- `examples/fast_texture_gen_multiview.py` from the official Hunyuan3D-2 repo
- The official `gradio_app.py` as API documentation
- The `tencent/Hunyuan3D-2mv` HuggingFace model tree

### 4.3 Hunyuan3D-2mv Repo

Research 2 is the only document that references the `tencent/Hunyuan3D-2mv` HuggingFace repo and its model variants:

- `hunyuan3d-dit-v2-mv` (standard)
- `hunyuan3d-dit-v2-mv-fast`
- `hunyuan3d-dit-v2-mv-turbo`

### 4.4 Public vs Internal API Understanding

Research 2 draws a clear distinction:

| API surface | Status per Research 2 |
|---|---|
| `from_pretrained` → returns mesh | **Public, stable** |
| `output_type="latent"` → separate VAE decode | **Internal, unstable** |
| `ShapeVAE` direct usage | "Only as conceptual pseudocode" |

This directly challenges the existing review's preference for Research 1's split latent flow, which Research 2 considers "internal / unstable API territory."

---

## 5. CORRECTIONS TO EXISTING REVIEW

The following verdicts from the original comparison review (`hunyuan3d_document_comparison.md`) should be reconsidered in light of Research 2:

### 5.1 Multiview Geometry Verdict — REVISED

**Original verdict (Multiview Geometry section):**
> "Both correctly identify it as dormant; Research adds the runtime fallback + warning" — MATCH

**Revised verdict:** Research 2 demonstrates that multiview geometry is NOT dormant in the broader Hunyuan3D ecosystem — it is functional via the `Hunyuan3D-2mv` repo with `from_pretrained`. The dormancy is specific to the **ComfyUI wrapper's** 2.1 multiview node and the missing `dit_config_2_1_mv.yaml`. Both the Tutorial and Research 1 correctly describe the wrapper situation, but neither identifies the working v2.0 multiview path that Research 2 uses.

### 5.2 Stage D Verdict — NUANCED

**Original verdict:**
> "Research is correct" for implementing `output_type="latent"` with the split flow.

**Nuance from Research 2:** Research 2 explicitly recommends AGAINST the latent split flow for production, calling it "internal / unstable API territory." The public API returns a mesh directly from `from_pretrained`. Research 1's split flow is technically more flexible but potentially brittle across upstream updates. The Tutorial's `output_type="trimesh"` in the runnable code may actually be closer to the recommended public API approach, even though its prose contradicts it.

### 5.3 Stage E Verdict — NUANCED

**Original verdict:**
> "Research is correct" that VAE is a separate checkpoint.

**Nuance from Research 2:** Research 2 agrees the VAE exists as a separate artifact, but its recommended approach (use `from_pretrained` which handles everything internally) makes the question moot for production use. The separate VAE loading only matters if you're using the internal latent flow that Research 2 discourages.

### 5.4 Requirements.txt — ADDITIONAL CORRECTION

**Original verdict:** Research 1's requirements are more carefully curated.

**Additional finding:** Research 2 identifies `xatlas==0.0.10` as the correct pin per official 2.1 requirements. Both the Tutorial (`0.0.9`) and Research 1 (`0.0.9`) should be updated.

---

## 6. Cross-Document Summary Table

| Feature | Tutorial | Research 1 | Research 2 |
|---|---|---|---|
| **Shape model** | `hunyuan3d-dit-v2-1` (single-image) | `hunyuan3d-dit-v2-1` (single-image) | `hunyuan3d-dit-v2-mv` (four-view) |
| **Shape loading** | `from_single_file` or `from_pretrained` | `from_single_file` only | `from_pretrained` only |
| **Latent API stance** | Prose says latent; code says trimesh | Implements latent split flow | Recommends against latent; use public mesh API |
| **Delight step** | Absent | Absent | Present (`Light_Shadow_Remover`) |
| **Background composite** | White | White | Gray (127) |
| **Paint config constructor** | `(max_num_view, resolution)` | `(max_num_view, resolution)` | `(resolution, camera_azims, camera_elevs, view_weights, ortho_scale, texture_size)` |
| **xatlas version** | `0.0.9` | `0.0.9` | `0.0.10` |
| **Mesh normalization** | Absent | Absent | Present |
| **Multiview geometry** | Dormant / documented only | Dormant / runtime fallback | **Active primary path** |
| **File architecture** | Single file | Single file | Multi-file `src/` package |
| **Inference context** | `@torch.no_grad()` | `@torch.no_grad()` | `torch.inference_mode()` |
| **multiviewDiffusionNet params** | No `num_steps`/`guidance_scale`/`seed` | Passes all three | Passes all three |
| **2mv repo referenced** | No | No | Yes (`tencent/Hunyuan3D-2mv`) |
| **Fast/turbo variants** | Not mentioned | Not mentioned | Documented |
| **Intermediate output org** | Flat `intermediates/` | Flat `outputs/debug/` | Numbered stage dirs (`01_`–`09_`) |

---

## 7. Recommendations

### For a production standalone pipeline:

1. **Use Research 2's modular architecture** — the multi-file `src/` package structure is easier to maintain and debug.
2. **Consider Research 2's delight step** — it adds meaningful quality for real-world photographs.
3. **Use the gray background composite** from Research 2 rather than white from the other documents.
4. **Use `xatlas==0.0.10`** per Research 2's identification of the official pin.
5. **Choose your shape pipeline deliberately:**
   - For **single-image** input: Use the Tutorial/Research 1 path with `hunyuan3d-dit-v2-1`.
   - For **four-view** input: Use Research 2's path with `hunyuan3d-dit-v2-mv`.
6. **Use `from_pretrained`** as Research 2 recommends — it has the most stable API surface.
7. **Pass `num_steps`, `guidance_scale`, and `seed`** to the multiview diffusion model (confirmed by both Research 1 and Research 2).
8. **Include mesh normalization** after postprocessing (Research 2's `normalize_mesh`).
9. **Use `torch.inference_mode()`** instead of `@torch.no_grad()` for all inference stages.

### For the Hunyuan3DPaintConfig constructor:

Research 2's constructor signature needs verification against the actual source code. If the constructor does accept `camera_azims`, `camera_elevs`, `view_weights`, `ortho_scale`, and `texture_size` as named parameters, this is cleaner than the attribute-mutation pattern used by the Tutorial and Research 1. If Research 2's signature is aspirational or version-specific, the mutation pattern remains the safe fallback.

---

## 8. Open Questions

1. **Which `Hunyuan3DPaintConfig` constructor is correct?** Research 2 uses a different signature from the other two documents. Only one can be correct for a given version of the upstream code.
2. **Is `xatlas==0.0.10` actually required?** Research 2 claims the official 2.1 requirements pin this version. This should be verified against the upstream `requirements.txt`.
3. **Does the delight model require separate download?** Research 2 treats it as optional but doesn't provide a download script for the `light_remover_ckpt_path` directory.
4. **How does Research 2's 2mv shape model interact with the 2.1 paint model?** Research 2 uses a v2.0 shape model with what appears to be a v2.1 paint pipeline. Cross-version compatibility should be validated.
5. **Is `compose_over_gray` provided by the upstream code?** Research 2 imports it from `.preprocess` (its own module), suggesting it may be a custom implementation rather than an upstream utility.
