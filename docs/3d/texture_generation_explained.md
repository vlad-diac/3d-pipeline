# Texture Generation — How It Works and How to Improve It

> A step-by-step explanation of how this pipeline turns one or more input photographs
> into a fully textured PBR GLB, and a catalogue of concrete improvements you can make.

---

## 1. The Big Picture

The pipeline solves a hard problem: a 3D mesh has no colour information — it is just
geometry. To make it look like the real object you photographed, you need to *paint* it.
The strategy used here is **multiview diffusion + UV baking**:

1. **Generate views** — use a diffusion model to "imagine" what the object looks like from 6
   angles it was never photographed from.
2. **Project back** — mathematically reproject those 6 views onto the UV texture atlas of
   the mesh (like wrapping a gift with colour photographs).
3. **Fill the gaps** — any UV texel not seen by any camera is filled in by inpainting.

The output is a standard PBR (Physically Based Rendering) material with an **albedo map**
(base colour) and a **metallic-roughness map** (surface material properties), packed into a
GLB file readable by any modern 3D viewer or game engine.

---

## 2. Full Pipeline Flow

```
Input image(s)
    │
    ▼
[A] Load & validate images  ←  preprocess.py: load_image_rgba()
    │
    ▼
[B] Background removal + composite  ←  preprocess.py: maybe_remove_background()
    │  ┌─────────────────────────────────────────────────────────┐
    │  │  White BG composite  →  shape model input               │
    │  │  Gray  BG composite  →  paint model reference           │
    │  └─────────────────────────────────────────────────────────┘
    ▼
[C+D] Shape generation  ←  mesh_generate.py
    │  Hunyuan3D-DiT v2.1 (single-image)  OR  v2.0-2mv (4-view)
    │  Output: raw trimesh.Trimesh (up to ~500k faces)
    ▼
[E] Mesh postprocess  ←  mesh_postprocess.py
    │  Remove floaters, degenerate faces, decimate → 200k faces, normalize
    ▼
[F] UV unwrap  ←  render_multiview.py: uv_unwrap_mesh()
    │  xatlas packs all faces into a [0,1]² texture atlas
    ▼
[G] 6-camera config  ←  config.py: CameraConfig
    │  Front(0°), Right(90°), Back(180°), Left(270°), Top(90°el), Bottom(-90°el)
    ▼
[H] Paint pipeline init  ←  render_multiview.py: PaintPipeline
    │  MeshRender (CUDA rasterizer)  +  ViewProcessor
    ▼
[I] Render conditioning maps  ←  render_multiview.py: render_conditioning_maps()
    │  6 normal maps  (surface orientation, RGB = Nx,Ny,Nz)
    │  6 position maps (world-space coords, normalized RGB)
    ▼
[J] Delight reference  ←  paint_multiview.py: delight_reference()   [optional]
    │  Light_Shadow_Remover strips baked-in highlights/shadows from front view
    ▼
[K] Multiview diffusion  ←  paint_multiview.py: MultiviewDiffusionNet
    │  HunyuanPaintPipeline (SD-based UNet conditioned on normals + positions)
    │  Output: 6 albedo views  +  6 metallic-roughness views
    ▼
[L] Upscale views  ←  paint_multiview.py: upscale_views()
    │  Lanczos (default)  OR  RealESRGAN 4× (higher quality)
    │  Target: render_size (default 2048 px per view)
    ▼
[M] UV bake  ←  bake_texture.py: bake_multiview_textures()
    │  Cosine-weighted back-projection: each texel accumulates colour
    │  from all views that can see it, weighted by facing angle
    │  Output: albedo texture [H,W,3]  +  MR texture [H,W,3]  +  coverage masks
    ▼
[N] Inpaint  ←  inpaint_texture.py: inpaint_textures()
    │  Pass 1: vertex-aware C++ extension fills seam gaps along mesh edges
    │  Pass 2: OpenCV Navier-Stokes fills remaining 2D holes
    ▼
[O] Export  ←  export_glb.py: export_textured_mesh()
    │  set_texture() + set_texture_mr() → save_mesh() → OBJ + JPEG textures
    │  create_glb_with_pbr_materials() → PBR GLB
    ▼
  output.glb
```

---

## 3. Stage-by-Stage Explanation

### Stage B — Background Removal and Compositing

Two different composites are made from the same RGBA image:

| Purpose | Background colour | Why |
|---------|------------------|-----|
| Shape model input | White (255, 255, 255) | DiT was trained on white-background synthetic images |
| Paint model reference | Gray (127, 127, 127) | Neutral mid-grey reduces directional lighting bias baked into the texture |

If the input photo has a real background, `rembg` (ONNX U-2-Net) extracts the foreground
alpha mask before compositing. The `--remove-bg` flag enables this.

---

### Stage I — Normal Maps and Position Maps

Before the diffusion model can paint anything, the renderer rasterizes the UV-unwrapped
mesh from each of the 6 camera views and produces two conditioning images per view:

**Normal map** — each pixel's RGB encodes the surface normal direction (Nx, Ny, Nz).
Blue-ish pixels face the camera directly; red/green pixels are angled surfaces. This tells
the paint model where edges and creases are.

**Position map** — each pixel's RGB encodes the world-space 3D position of that surface
point (normalized to [0,1]). This gives the model spatial consistency — the same physical
point on the mesh maps to the same colour in every view's position map, so the diffusion
model can match colours across views.

Together these two inputs are what the diffusion model uses instead of classical texture
coordinates — they act as a geometric conditioning signal.

---

### Stage J — Delighting (Optional)

A real-world photograph contains highlights and shadows that are specific to the lighting
conditions when the photo was taken. If you bake those into the texture, your 3D object
will always look like it was lit from the same direction — even when placed in a different
scene with different lighting.

`Light_Shadow_Remover` (from the Hunyuan3D-2.0 codebase) runs an intrinsic image
decomposition network that separates appearance (albedo) from illumination, returning a
flat-lit version of your photo. This flat-lit image is what the diffusion model uses as its
appearance anchor.

This step is skipped if the delight model checkpoint is not downloaded, falling back to the
plain gray composite.

---

### Stage K — Multiview PBR Diffusion (the core step)

This is where the actual texture is "imagined". The `HunyuanPaintPipeline` is a
**Stable Diffusion-based UNet** fine-tuned for 3D texture generation. Its architecture has
several key characteristics:

**Inputs to the model:**
- **Reference image** (1×): the delighted front-view photo — the appearance anchor
- **Normal maps** (6×): geometric conditioning from each camera
- **Position maps** (6×): spatial consistency conditioning
- **DINOv2 features** (optional): extracted from the reference image and injected into the
  UNet cross-attention layers, providing stronger appearance conditioning than the raw pixels

**What the model generates:**
- **6 albedo images** — the base colour from each camera view
- **6 metallic-roughness images** — the PBR material from each camera view
  (R = metallic, G = roughness, B = unused, as per glTF PBR convention)

All 12 images are generated jointly in a single forward pass — the attention layers across
the batch mean the model sees all views simultaneously and enforces cross-view consistency.

**Scheduler**: UniPCMultistep with trailing timestep spacing. Good quality at 10–15 steps.

**Key parameters:**
| Parameter | Default | Effect |
|-----------|---------|--------|
| `paint_steps` | 10 | More steps = higher quality, slower |
| `paint_guidance_scale` | 3.0 | Higher = colours closer to reference, less creative fill |
| `view_size` | 512 | Diffusion resolution (model was trained at 320) |
| `seed` | 42 | Controls stochastic sampling |

---

### Stage M — Cosine-Weighted UV Baking

The 6 generated view images need to be mapped back into the UV texture atlas. This is
done by the CUDA `custom_rasterizer` extension.

For each texel in the UV atlas:

1. **Visibility test** — rasterize from each of the 6 cameras to determine which cameras can
   actually see this surface point (not occluded by other geometry).

2. **Colour sampling** — for each visible camera, project the 3D surface point back into
   that camera's 2D view image and sample the colour using bilinear interpolation.

3. **Weight computation**:
   ```
   weight = base_view_weight × cos(θ)^bake_exp
   ```
   - `base_view_weight` is the per-camera weight from `CameraConfig.weights`
     (front=1.0, back=0.5, sides=0.1, top/bottom=0.05)
   - `θ` is the angle between the surface normal and the camera ray
   - `bake_exp=4` means nearly face-on surfaces get much higher trust than grazing angles

4. **Blending** — all weighted colour contributions are summed and divided by total weight.
   Texels with zero total weight are marked as uncovered in the trust mask.

The result is two textures (`texture_albedo`, `texture_mr`) and two binary masks
(`mask_albedo`, `mask_mr`). Typically 85–95% of the UV atlas is covered.

---

### Stage N — Two-Pass Texture Inpainting

The uncovered texels (UV seams, pole regions, occluded back faces) must be filled so the
final texture has no black holes.

**Pass 1 — Vertex-aware inpainting**: The `meshVerticeInpaint` C++ extension walks along
mesh edges from covered vertices to uncovered neighbours, propagating colour through
adjacency. This respects the 3D topology and prevents colour bleeding across UV chart
boundaries — seam edges are filled correctly because the extension knows which vertices
are geometrically adjacent even if they are far apart in UV space.

**Pass 2 — OpenCV inpainting**: Any remaining 2D holes in the UV image are filled using
the Navier-Stokes PDE solver (`cv2.INPAINT_NS`). This is a purely 2D image operation that
grows colour inward from the hole boundary. It works well for small residual gaps.

---

### Stage O — GLB Export

The final assembly:

1. The renderer loads the same UV-unwrapped mesh (it must be the exact same instance —
   the internal GPU buffer must match the UV coordinates used during baking).
2. `set_texture()` and `set_texture_mr()` bind the inpainted textures.
3. `save_mesh()` writes an OBJ file and 3 JPEG files (albedo, metallic, roughness).
4. `create_glb_with_pbr_materials()` assembles a glTF 2.0 binary with a
   `pbrMetallicRoughness` material referencing all three texture maps.

---

## 4. Where Each Model Lives

| Model | File | Size | Role |
|-------|------|------|------|
| Hunyuan3D-DiT v2.1 | `models/hunyuan3d-dit-v2-1/` | ~7.4 GB | Shape generation (single-image) |
| Hunyuan3D-DiT 2mv | `models/hunyuan3d-dit-v2-mv/` | ~7 GB | Shape generation (four-view) |
| Hunyuan3D-VAE v2.1 | `models/hunyuan3d-vae-v2-1/` | ~656 MB | Latent → mesh decoder |
| **HunyuanPaintPBR v2.1** | `models/hunyuan3d-paintpbr-v2-1/` | ~4 GB | **Texture diffusion (the one that generates colours)** |
| DINOv2-Giant | `models/dinov2-giant/` | ~4.4 GB | Appearance conditioning for paint UNet |
| Light_Shadow_Remover | `models/delight/` | ~2 GB | Delight preprocessing |
| RealESRGAN x4plus | `models/RealESRGAN_x4plus.pth` | ~64 MB | View upscaling |

The model you would most want to replace is **HunyuanPaintPBR v2.1**. It is entirely
swappable — see Section 5.

---

## 5. Improvements You Can Make

### 5.1 Replace the Texture Diffusion Model

The biggest lever for texture quality. The `MultiviewDiffusionNet` class in
`src/paint_multiview.py` is the integration point.

**Option A — TEXTure / Text2Tex**
These models condition on text prompts in addition to geometry. Useful when your input
image has bad lighting or is not representative of all sides.

> Swap `MultiviewDiffusionNet.__call__` to call TEXTure's pipeline, passing normal maps
> as ControlNet conditioning and a text description as the prompt.

**Option B — SyncDreamer / Zero123++**
These are diffusion models specifically trained to generate geometrically consistent
novel views from a single photo. You could use one of them instead of HunyuanPaint to
generate the 6 views, then bake those directly — the rest of the pipeline (bake, inpaint,
export) stays identical.

> In `main.py` / `_run_paint()`, replace the `MultiviewDiffusionNet(cfg)` block with a
> call to a SyncDreamer or Zero123++ wrapper.

**Option C — Wonder3D**
Wonder3D generates multi-view RGB + normal images jointly from a single input image. Its
normal outputs could replace both the diffusion-generated views and the rasterized normal
maps, potentially improving view consistency.

**Option D — Upgrade to HunyuanPaint v3 (when available)**
The architecture in `MultiviewDiffusionNet` is loaded via `DiffusionPipeline.from_pretrained`
with a `custom_pipeline` path. To upgrade, download the new model weights and update
`cfg.models_dir / "hunyuan3d-paintpbr-v2-1"` to point to the new directory. The scheduler
and VAE optimization code can stay the same.

---

### 5.2 Improve the Reference Image (Delight Step)

The delight step has the biggest impact on real-world photographs. Options:

**IC-Light** (ByteDance, 2024) — a diffusion-based relighting model that can relight your
object photo under arbitrary target lighting, not just remove it. Running it before the
paint pipeline would give a cleaner, relightable texture.

> Add an `ic_light_reference()` function in `src/paint_multiview.py` alongside
> `delight_reference()` and select it via a `cfg.delight_mode = "ic-light"` config flag.

**Intrinsic decomposition (Intrinsic Images in the Wild)** — a different approach that
separates an image into albedo + shading layers using a deep network. The albedo layer is
used as the paint model reference. More principled than the current shadow removal approach.

---

### 5.3 Improve UV Unwrapping

The current `xatlas` UV unwrap has no control over seam placement. Two improvements:

**Better seam placement** — xatlas has options for specifying chart boundary constraints.
Placing seams along geometrically natural boundaries (edges with high dihedral angle,
silhouette edges) reduces visible texture seams. Expose `xatlas.PackOptions` and
`xatlas.ChartOptions` in `uv_unwrap_mesh()`.

**Neural UV parametrization (NvDiffRast-based)** — learnable UV maps optimized to minimize
distortion. Requires differentiable rasterization but produces significantly better UV
packing for organic shapes (characters, animals). NvDiffRast or Kaolin provide this.

---

### 5.4 Improve the Baking Step

**Higher view count** — the current `CameraConfig` uses 6 views. Adding 4 diagonal
cameras (azimuth 45°, 135°, 225°, 315° at elevation 30°) covers shoulders and concave
regions better. Just add entries to `CameraConfig.azimuths`, `.elevations`, and `.weights`
in `src/config.py` — the rest of the pipeline scales automatically.

```python
# Add 4 diagonal views at 30° elevation
CameraConfig(
    azimuths   = [0, 90, 180, 270, 0, 180, 45, 135, 225, 315],
    elevations = [0, 0,  0,   0,  90, -90, 30, 30,  30,  30],
    weights    = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2],
)
```

**Poisson blending at seams** — instead of the current cosine-weighted average, use Poisson
image editing to blend colour at UV chart boundaries. This eliminates the hard colour seam
that sometimes appears where xatlas charts meet.

**Increase `bake_exp`** — the cosine exponent controls how much weight is given to
face-on views. Higher values (6–8 instead of 4) further reduce colour bleed from
grazing-angle views. Change `bake_exp=4` in `render_multiview.py: PaintPipeline.__init__`.

---

### 5.5 Better Upscaling

**SwinIR / HAT** — more recent single-image super-resolution models that outperform
RealESRGAN x4plus on texture-like content (fine detail, fabric, skin). Swap the upscaler
in `upscale_views()` in `src/paint_multiview.py`.

**Diffusion-based upscaling (StableSR / SUPIR)** — for extreme quality, these models
hallucinate fine detail consistent with the overall texture. Slower but significantly
sharper output at high `texture_size` values (8192+).

---

### 5.6 Add a Texture Refinement Pass

After baking + inpainting, the UV texture may still have inconsistencies at chart
boundaries (slight colour shifts where different views bled in). A refinement pass can
fix this:

**Texture-space diffusion refinement** — run a second diffusion pass where the baked
texture is used as the starting latent (img2img with low denoising strength ~0.3).
The model smooths out blotchy areas without regenerating the whole texture.

**Texture harmonization** — an existing technique from image editing that makes
colour-pasted regions look like they belong to the same image. Apply per-chart with the
boundary as the mask.

To add this, insert a new step between `inpaint_textures()` and `export_textured_mesh()`
in `main.py`:

```python
# After _run_inpaint():
if cfg.texture_refinement:
    refined_albedo = run_texture_refinement(refined_albedo, cfg)
```

---

### 5.7 Texture Resolution

The current default is `texture_size=4096`. Common improvements:

| texture_size | Use case | VRAM impact |
|-------------|----------|-------------|
| 1024 | Preview / mobile | Minimal |
| 2048 | Standard game asset | Moderate |
| 4096 | High-quality asset (default) | Significant |
| 8192 | Hero asset / close-up | Requires A100/H100 |

Increasing `texture_size` requires increasing `render_size` proportionally (always keep
`render_size >= texture_size`) to avoid aliased baking. Change both in `PipelineConfig`
or via CLI: `--texture-size 8192 --render-size 8192`.

---

### 5.8 Per-Object Text Prompt Conditioning

The current `MultiviewDiffusionNet.__call__` always passes `prompt="high quality"`.
HunyuanPaintPipeline supports arbitrary text prompts. Passing a description of the object
can significantly improve texture plausibility for views not visible in the input photo:

```python
# src/paint_multiview.py in MultiviewDiffusionNet.__call__
outputs = self.pipeline(
    [input_img],
    prompt=cfg.paint_prompt,   # e.g. "a wooden chair with dark stain finish"
    num_inference_steps=cfg.paint_steps,
    ...
)
```

Expose this via `PipelineConfig.paint_prompt: str = "high quality"` and
`main.py --prompt "..."`.

---

## 6. Quality vs. Speed Tradeoffs

| Goal | Change | Impact |
|------|--------|--------|
| Faster iteration | `--paint-steps 5` | ~2× faster, slightly noisier colours |
| Faster iteration | `--texture-size 1024` | Much less VRAM + time |
| Better back/side | Add 4 diagonal views to `CameraConfig` | Richer coverage |
| Sharper texture | `--use-realesrgan` | +20–30% sharper at same resolution |
| No baked lighting | `--use-delight` (ensure model downloaded) | Cleaner PBR material |
| More creative fill | Lower `--paint-guidance 1.5` | Model adds more detail to unseen areas |
| Colour fidelity | Raise `--paint-guidance 5.0` | Closer to input photo colours |
| Larger mesh detail | `--octree-resolution 512` | More geometry, more VRAM |

---

## 7. Module Map

| Source file | Stage(s) | Can you swap it? |
|-------------|----------|-----------------|
| `src/preprocess.py` | A, B | Yes — swap background removal model |
| `src/mesh_generate.py` | C, D | Yes — but shape pipeline API must match |
| `src/mesh_postprocess.py` | E | Yes — any decimation library |
| `src/render_multiview.py` | F, H, I | Partially — UV unwrap is swappable, renderer less so |
| `src/paint_multiview.py` | **J, K, L** | **Yes — this is the main swap point for texture quality** |
| `src/bake_texture.py` | M | Partially — bake logic tied to ViewProcessor |
| `src/inpaint_texture.py` | N | Yes — swap in any inpainting method |
| `src/export_glb.py` | O | Yes — export format, PBR assembler |
| `src/config.py` | all | Add params here to expose new knobs |
