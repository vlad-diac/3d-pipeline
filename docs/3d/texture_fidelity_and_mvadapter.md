# Why Textures Don't Match Your Photo — and How to Fix It with MV-Adapter

---

## 1. The Real Problem

When you feed a realistic photograph, the pipeline produces a texture that looks like a
plausible version of your object but not like the actual photo you provided.
The root cause is **not** baking, UV unwrapping, or inpainting. It is the **texture
diffusion step (Stage K)**.

### What the current pipeline actually does

`HunyuanPaintPipeline` is a **Stable Diffusion fine-tune** trained to generate
plausible-looking multiview PBR textures. It is a *generative* model, not a
*projection* model. Its forward pass looks like this:

```
Inputs:
  ┌─ reference_rgb       (1 image: your front view, gray-composited + delighted)
  ├─ normal_maps         (6 × rasterized normals from the mesh)
  ├─ position_maps       (6 × world-space coordinate maps from the mesh)
  └─ DINOv2 features     (extracted from the reference — the "style" signal)

Outputs:
  ┌─ albedo views [0..5]  ← diffusion-generated colours per view
  └─ MR views [0..5]      ← diffusion-generated metallic-roughness per view
```

The model uses the normals + positions to understand geometry, the front view + DINOv2
features as an appearance anchor, and **fills in all views stochastically based on its
training distribution**. For a toy chair or synthetic object it works well because the
training distribution covers that domain. For a real photograph of a ship, crane vessel,
or any complex textured object it:

- **Loses fine details** (rivets, logos, weathering, rust patterns) because the SD-level
  resolution at 320 px can't encode them
- **Invents the sides and back** based on what a generic version of the object looks like,
  not what your specific object looks like
- **Bakes in a lighting-corrected but stylistically shifted appearance** because the
  `delight_reference()` step and gray composite change the colour temperature and contrast
  before the image ever reaches the paint model
- **Loses colour fidelity on the front view itself** because the diffusion process is
  stochastic — even with `guidance_scale=3.0`, the output is not a deterministic copy of
  the reference

This is the gap: the model **cannot be made to copy your photo**. It is a conditional
generator, not a renderer.

---

## 2. How the Canonical Multiview Research Applies

The `canonical-multiview-generation.md` research was written to solve a different
surface problem (bad canonical view assignment for the shape model), but it describes the
**exact tool that fixes the texture fidelity problem**: **MV-Adapter i2mv**.

### What MV-Adapter actually does

MV-Adapter is an adapter trained on top of **SDXL** that converts it into a
**multiview image generator**. Given one reference photo and camera azimuths, it
produces geometrically consistent views of that exact object — using the real photo as
a strong appearance anchor at full SDXL resolution (768 px). Crucially:

- The reference photo is not averaged, delighted, or composited to gray first
- The output views look like the reference photo, just from different angles
- Depth and canny controls can stabilize hull geometry and thin structures
- The `mvadapter_i2mv_sdxl_beta.safetensors` checkpoint was specifically designed for
  `front / right / back / left` canonical 4-view generation

These views are photorealistic projections of your object, not hallucinations from a
3D-object training distribution.

### The fundamental shift in pipeline logic

| | Current pipeline | With MV-Adapter |
|--|--|--|
| Albedo source | HunyuanPaint generates views from normals | MV-Adapter generates views from the real photo |
| Resolution | 320–512 px diffusion space | 768 px SDXL space |
| Colour fidelity | Stochastic, guided by DINOv2 features | Directly anchored to reference photo |
| Side/back content | Hallucinated from 3D training priors | Extrapolated from the actual image appearance |
| MR channel | Fully generated (no photo reference) | Still generated — MR has no photo equivalent |

---

## 3. Concrete Integration Paths

Three paths, ordered by implementation effort and quality gain:

---

### Path 1 — MV-Adapter as Pre-Step for Shape (Low effort, partial gain)

**What it changes:** Shape fidelity only. Texture still comes from HunyuanPaint.

```
Your photo
    │
    ▼
MV-Adapter i2mv  →  front / right / back / left  →  Hunyuan3D-2mv shape
                                                           │
                                                       (existing pipeline)
                                                       UV unwrap → HunyuanPaint → bake → GLB
```

You already support this through `--use-multiview-shape` + `--front/--left/--right/--back`.
The canonical multiview script from the research would generate those 4 inputs
automatically from a single photo.

**Gain:** Better mesh because Hunyuan3D-2mv gets correctly-oriented canonical views
instead of a single ambiguous photo. Texture is still approximate.

---

### Path 2 — MV-Adapter Views as Direct Bake Inputs (High effort, large quality gain)

**What it changes:** Albedo texture quality. This is the main fix for "texture looks
nothing like the photo."

**Core idea:** Skip HunyuanPaint for the albedo channel. Instead, use MV-Adapter's
photo-realistic views directly as bake source images.

```
Your photo
    │
    ├──► MV-Adapter i2mv  →  4 views (front/right/back/left) at 768 px
    │                                  │
    │                                  ▼
    │         Upscale to render_size (Lanczos or RealESRGAN)
    │                                  │
    │                                  ▼
    └──────────────────────────► UV bake (cosine-weighted projection)
                                         │
                                         ▼
                               HunyuanPaint (MR channel only)
                                         │
                                         ▼
                               Inpaint → Export GLB
```

For the MR (metallic-roughness) channel there is no photographic reference — the physical
material properties (how metallic or rough a surface is) are not encoded in a photograph.
HunyuanPaint is still the best source for MR because it was trained on PBR materials.

**Key implementation change in `main.py`:**

```python
# Replace _run_paint() with:

def _run_mvadapter_albedo(front_rgba, cfg, intermediates_root):
    """Generate albedo views using MV-Adapter instead of HunyuanPaint."""
    from src.canonical_multiview import generate_canonical_views  # new module
    
    albedo_views = generate_canonical_views(
        reference=front_rgba,
        azimuths=[0, 90, 180, 270],
        cfg=cfg,
    )
    # 4 views from MV-Adapter → add top/bottom placeholders or skip those cameras
    return albedo_views

def _run_hunyuanpaint_mr_only(front_rgba, normal_maps, position_maps, cfg):
    """Run HunyuanPaint but only use the MR output, discard albedo."""
    from src.paint_multiview import MultiviewDiffusionNet, delight_reference
    reference = delight_reference(front_rgba, cfg)
    mvd = MultiviewDiffusionNet(cfg)
    paint_out = mvd(reference, normal_maps, position_maps, cfg)
    mvd.unload()
    return paint_out["mr"]   # only keep MR
```

**Camera config adjustment** — MV-Adapter generates 4 views (front/right/back/left),
but the bake uses 6. Two options:

1. **Use 4 cameras for baking** — adjust `CameraConfig` to use only the 4 cardinal views.
   Top and bottom texels will be uncovered and inpainted from neighbours (usually fine for
   ships/products where top and bottom are small or flat).

2. **Add top/bottom from HunyuanPaint** — run HunyuanPaint for all 6 views, use
   MV-Adapter for views 0–3 (albedo) and HunyuanPaint for views 4–5 (top/bottom only).

---

### Path 3 — Hybrid: MV-Adapter + HunyuanPaint Blended (Medium effort, best quality)

**What it changes:** Maximum colour fidelity and material quality.

The insight: MV-Adapter is good at appearance fidelity but is an RGB image generator —
it cannot produce PBR-correct metallic and roughness values. HunyuanPaint generates
plausible PBR but loses photorealism. Blending them is the best of both.

**For albedo:** Use MV-Adapter as the primary source, weighted heavily on the front view.

**For MR:** Use HunyuanPaint. The MR generation is where HunyuanPaint actually adds real
value — the model was trained on PBR materials and produces reasonable metallic/roughness
maps even for unseen objects.

**For top/bottom views:** Let HunyuanPaint fill those in — MV-Adapter doesn't see them
and they're usually small surface areas anyway.

```python
# Bake weights when mixing sources:
CameraConfig(
    # Only bake from the 4 MV-Adapter views (drop top/bottom or use lower weight)
    azimuths   = [0, 90, 180, 270],
    elevations = [0, 0,  0,   0 ],
    weights    = [1.0, 0.4, 0.6, 0.4],   # front still dominant
)
```

---

## 4. What the `canonical_multiview.py` Script Does (from the Research)

The research document provides a near-complete implementation you can drop in as a new
`src/canonical_multiview.py` module. Its key components map directly to what you need:

| Research component | What it produces | How it helps texture |
|---|---|---|
| `remove_bg_birefnet()` | Clean RGBA foreground | Better MV-Adapter anchor; no background contaminating side views |
| `center_pad_resize()` | Centered 768×768 RGB on gray | Matches MV-Adapter training distribution |
| `build_depth_map()` (DPT/MiDaS) | Depth control image | Stabilizes hull and crane geometry across 4 views |
| `build_canny_map()` | Edge control image (weak) | Preserves thin structures (masts, rails) |
| `build_plucker_controls()` | Camera-aware control tensor | Tells MV-Adapter exactly which azimuth each view should be |
| `generate_canonical_views()` | front/right/back/left images at 768 px | **Replaces HunyuanPaint albedo output** |

The only piece the research doesn't include is the **integration into the existing bake
pipeline** (the `bake_multiview_textures()` call). That is the only new code you need.

---

## 5. Why Realistic Images Suffer More than Synthetic Images

| Factor | Synthetic/CG object | Real photograph |
|---|---|---|
| Background | Already removed, clean alpha | Cluttered — harbour, water, sky |
| Lighting | Even, studio-like | Complex directional — creates DINOv2 confusion |
| Texture detail | Flat, low-frequency | High-frequency details HunyuanPaint can't encode at 320 px |
| Object coverage | Usually 1 view covers the whole object | Side occluded by cranes, containers, quay |
| HunyuanPaint training domain | Trained on clean synthetic 3D objects | Real photographs are OOD (out of distribution) |

The gray composite and delight step **deliberately discard lighting information** from the
reference image. For a clean CG render that's fine — there's no real lighting to preserve.
For a photograph of a real ship, you're discarding real surface colour variation (rust
streaks, waterline marks, weathering) because the model can't distinguish them from
lighting. MV-Adapter does not do this — it runs at the full SDXL resolution with the
original photograph as conditioning.

---

## 6. Implementation Plan

The minimal change to improve texture fidelity for real photos:

### Step 1 — Create `src/canonical_multiview.py`

Extract the MV-Adapter generation logic from the research skeleton into a proper module:

```
src/
  canonical_multiview.py    ← new: wraps MV-Adapter i2mv pipeline
```

Key functions needed:
- `load_mvadapter_pipe(cfg)` — load SDXL + MV-Adapter beta checkpoint + optional ControlNets
- `generate_canonical_views(reference, azimuths, cfg)` → `List[PIL.Image]`
- `build_depth_map(img, device)` — DPT/MiDaS preprocessor
- `build_canny_map(img)` — opencv canny

### Step 2 — Add MV-Adapter config to `PipelineConfig`

```python
# src/config.py additions
use_mvadapter_albedo:   bool  = False
mvadapter_checkpoint:   str   = "mvadapter_i2mv_sdxl_beta.safetensors"
mvadapter_adapter_repo: str   = "huanngzh/mv-adapter"
mvadapter_depth_scale:  float = 0.5
mvadapter_canny_scale:  float = 0.2
mvadapter_steps:        int   = 50
mvadapter_guidance:     float = 3.0
```

### Step 3 — Add `--use-mvadapter-albedo` flag to `main.py`

```python
# In _run_paint(), branch on cfg.use_mvadapter_albedo:
if cfg.use_mvadapter_albedo:
    albedo_up = _run_mvadapter_albedo(front_rgba, cfg, intermediates_root)
    mr_up     = _run_hunyuanpaint_mr_only(front_rgba, normal_maps, position_maps, cfg)
else:
    # existing path
    albedo_up, mr_up = _run_paint(front_rgba, normal_maps, position_maps, cfg, intermediates_root)
```

### Step 4 — Add to `download_models.py`

```python
# New model entries:
{
    "repo_id": "huanngzh/mv-adapter",
    "filename": "mvadapter_i2mv_sdxl_beta.safetensors",
    "local_dir": "models/mv-adapter",
},
{
    "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "local_dir": "models/sdxl-base-1.0",
},
{
    "repo_id": "madebyollin/sdxl-vae-fp16-fix",
    "local_dir": "models/sdxl-vae-fp16-fix",
},
{
    "repo_id": "Intel/dpt-hybrid-midas",      # depth ControlNet backend
    "local_dir": "models/dpt-hybrid-midas",
},
```

### Step 5 — Adjust `CameraConfig` for 4-view baking

```python
# In main.py when --use-mvadapter-albedo is set:
cfg.camera = CameraConfig(
    azimuths   = [0, 90, 180, 270],
    elevations = [0,  0,   0,   0],
    weights    = [1.0, 0.4, 0.6, 0.4],
)
```

---

## 7. Additional Improvement: Better Reference for HunyuanPaint

Even without MV-Adapter, there is a simpler improvement that helps:
**stop gray-compositing the reference image**. The gray composite was designed to remove
lighting bias, but for realistic photographs it also removes real surface colours. Try
passing the delight output directly without the gray composite, or skip delight entirely
and pass the original RGBA-over-white composite as the reference. This alone often
improves the front-face texture fidelity significantly.

```python
# src/paint_multiview.py: delight_reference()
# Option: skip gray composite, use white composite instead
reference = compose_over_white(front_rgba)   # instead of compose_over_gray
```

---

## 8. New Models Required

| Model | Size | Purpose |
|---|---|---|
| `mvadapter_i2mv_sdxl_beta.safetensors` | ~1 GB | MV-Adapter weights (4-view image-to-multiview) |
| `stabilityai/stable-diffusion-xl-base-1.0` | ~7 GB | SDXL backbone |
| `madebyollin/sdxl-vae-fp16-fix` | ~160 MB | Numerically stable SDXL VAE |
| `Intel/dpt-hybrid-midas` | ~400 MB | Depth map for depth ControlNet |
| `diffusers/controlnet-depth-sdxl-1.0` | ~2.5 GB | (Optional) depth ControlNet for structural stability |
| `diffusers/controlnet-canny-sdxl-1.0` | ~2.5 GB | (Optional) canny ControlNet for thin structures |

VRAM note: MV-Adapter on SDXL requires ~14 GB. The overall pipeline must run these two
heavy stages sequentially (MV-Adapter → free VRAM → HunyuanPaint for MR → free VRAM →
bake), not concurrently.

---

## 9. Summary

| Problem | Root cause | Fix |
|---|---|---|
| Texture looks nothing like the photo | HunyuanPaint is generative, not a renderer | Replace albedo with MV-Adapter views |
| Sides/back are generic-looking | Paint model hallucinates unseen views from training priors | MV-Adapter extrapolates from the actual photo |
| Fine details lost | HunyuanPaint runs at 320 px, SD resolution | MV-Adapter runs at 768 px SDXL resolution |
| Colour shift vs original photo | Gray composite + delight removes real surface colours | Pass white-composited image or test without delight |
| MR channel still approximate | No photographic reference for physical material properties | Keep HunyuanPaint for MR only — this is where it adds value |
