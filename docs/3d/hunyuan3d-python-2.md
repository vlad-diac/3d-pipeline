```python
    if not parts:
        return mesh
    return max(parts, key=lambda m: len(m.faces))


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    scale = float(np.max(bounds[1] - bounds[0]))
    if scale <= 1e-8:
        return mesh
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale)
    return mesh


def postprocess_mesh(mesh: trimesh.Trimesh, target_faces: int = 200_000) -> trimesh.Trimesh:
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
            pass

    mesh = normalize_mesh(mesh)
    mesh.remove_unreferenced_vertices()
    return mesh


def export_debug_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path
```

The public multiview shape route comes from the official 2.0 codebase plus the 2mv checkpoint repo; the official Gradio path supports multiview dictionary input, and the public pipeline signature accepts `dict` / `List[dict]` image input. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2/main/gradio_app.py))

### Advanced latent decode note

For the production script, use the public pipeline above. That is the path with the most stable API surface.

If you want to expose the **latent** explicitly and then decode it yourself through a VAE + marching cubes stage, you are stepping into internal / unstable API territory. The public examples return a mesh, while the wrapper source shows that a `ShapeVAE` exists in the internal stack. The public tutorials do **not** currently publish a stable “give me the latent tensor” API for 2mv shape generation. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/nodes.py))

Use this only as conceptual pseudocode:

```python
# PSEUDOCODE ONLY — internal API names vary by release

# latent = shape_transformer.sample(multiview_condition, steps=30, guidance=5.5)
# density_volume = shape_vae.decode(latent)
# verts, faces = marching_cubes(density_volume, level=0.0)
# mesh = trimesh.Trimesh(vertices=verts, faces=faces)
```

That is the right mental model for steps F and G, but I would not recommend pinning your standalone pipeline to internal latent APIs unless you are comfortable maintaining your own fork.

### `src/render_multiview.py`

This file handles steps J through L:

- UV unwrap
- build paint config
- render normal maps
- render position maps

```python
from __future__ import annotations

import torch

from .config import PipelineConfig, bootstrap_third_party

cfg = bootstrap_third_party()
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # noqa: E402
from hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap  # noqa: E402


def build_paint_config(cfg: PipelineConfig) -> Hunyuan3DPaintConfig:
    conf = Hunyuan3DPaintConfig(
        resolution=cfg.view_size,
        camera_azims=cfg.camera_azimuths,
        camera_elevs=cfg.camera_elevations,
        view_weights=cfg.camera_weights,
        ortho_scale=cfg.ortho_scale,
        texture_size=cfg.texture_size,
    )
    conf.device = cfg.device
    conf.render_size = cfg.render_size
    conf.texture_size = cfg.texture_size
    return conf


def prepare_uv_and_renderer(mesh, cfg: PipelineConfig):
    conf = build_paint_config(cfg)
    pipe = Hunyuan3DPaintPipeline(conf)

    uv_mesh = mesh_uv_wrap(mesh)
    pipe.load_mesh(uv_mesh)
    return pipe, uv_mesh


@torch.inference_mode()
def render_geometry_maps(pipe, cfg: PipelineConfig):
    normal_maps = pipe.view_processor.render_normal_multiview(
        cfg.camera_elevations,
        cfg.camera_azimuths,
        use_abs_coor=True,
    )
    position_maps = pipe.view_processor.render_position_multiview(
        cfg.camera_elevations,
        cfg.camera_azimuths,
    )
    return normal_maps, position_maps
```

The wrapper’s raw source shows the exact config fields you need here and the exact internal method calls used later by the ComfyUI nodes: `mesh_uv_wrap`, `render_normal_multiview`, `render_position_multiview`, and the dedicated candidate camera arrays. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/textureGenPipeline.py))

### `src/paint_multiview.py`

This file handles steps M through R:

- composite the reference over gray
- run Delight
- run multiview paint diffusion
- save generated texture views
- optionally upscale them

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import torch
from PIL import Image

from .config import PipelineConfig, bootstrap_third_party
from .preprocess import compose_over_gray

cfg = bootstrap_third_party()
from hy3dgen.texgen.utils.dehighlight_utils import Light_Shadow_Remover  # noqa: E402
from hy3dpaint.utils.multiview_utils import multiviewDiffusionNet  # noqa: E402


def load_delight_model(cfg: PipelineConfig):
    if not cfg.delight_model_dir.exists():
        return None

    delight_cfg = SimpleNamespace(
        device=cfg.device,
        light_remover_ckpt_path=str(cfg.delight_model_dir),
    )
    return Light_Shadow_Remover(delight_cfg)


@torch.inference_mode()
def delight_reference(front_rgba: Image.Image, cfg: PipelineConfig) -> Image.Image:
    reference = compose_over_gray(front_rgba, gray=127)
    delight = load_delight_model(cfg)

    if delight is None:
        return reference

    return delight(reference)


def ensure_multiview_model(pipe) -> None:
    if pipe.model is None:
        pipe.model = multiviewDiffusionNet(pipe.config)


@torch.inference_mode()
def generate_texture_views(
    pipe,
    reference_rgb: Image.Image,
    normal_maps: List[Image.Image],
    position_maps: List[Image.Image],
    cfg: PipelineConfig,
) -> Tuple[List[Image.Image], List[Image.Image]]:
    ensure_multiview_model(pipe)

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
    return result["albedo"], result["mr"]


def upscale_views(images: List[Image.Image], target_size: int) -> List[Image.Image]:
    if not images:
        return images
    if images[0].size == (target_size, target_size):
        return images

    return [
        img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        for img in images
    ]
```

Two technical details matter here.

First, the 2.0 dehighlight utility loads a Diffusers pipeline from `light_remover_ckpt_path`; that is why Delight is easiest to manage as a local downloaded folder. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2/main/hy3dgen/texgen/utils/dehighlight_utils.py))

Second, the wrapper’s multiview texture net uses the **first** input reference image as the DINO-conditioned appearance anchor and sends `input_images[0:1]` into the paint pipeline. That is the concrete reason to anchor the texture stage on the **front** view, even though geometry conditioning comes from six rendered normal/position views. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/utils/multiview_utils.py))

### `src/bake_texture.py`

This file handles steps S and T:

- resize generated multiview textures for baking
- bake them into UV space
- return the missing-texel masks

```python
from __future__ import annotations

from typing import List, Tuple

from PIL import Image

from .config import PipelineConfig


def resize_for_bake(images: List[Image.Image], render_size: int) -> List[Image.Image]:
    return [
        img.resize((render_size, render_size), Image.Resampling.LANCZOS)
        for img in images
    ]


def bake_textures(
    pipe,
    albedo_views: List[Image.Image],
    mr_views: List[Image.Image],
    cfg: PipelineConfig,
):
    albedo_views = resize_for_bake(albedo_views, cfg.render_size)
    mr_views = resize_for_bake(mr_views, cfg.render_size)

    tex_albedo, mask_albedo, tex_mr, mask_mr = pipe.bake_from_multiview(
        albedo_views,
        mr_views,
        cfg.camera_elevations,
        cfg.camera_azimuths,
        cfg.camera_weights,
    )
    return tex_albedo, mask_albedo, tex_mr, mask_mr
```

The wrapper’s standalone paint pipeline exposes `bake_from_multiview` as a first-class method, and its raw source shows that baking is driven by the same elevation / azimuth / weight arrays used for map rendering. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/textureGenPipeline.py))

### `src/inpaint_texture.py`

This file handles steps U and V:

- mesh-aware vertex inpainting
- OpenCV-based texture inpainting

```python
from __future__ import annotations

from .config import PipelineConfig


def inpaint_textures(
    pipe,
    tex_albedo,
    mask_albedo,
    tex_mr,
    mask_mr,
    cfg: PipelineConfig,
):
    refined_albedo, refined_mr = pipe.inpaint(
        tex_albedo,
        mask_albedo,
        tex_mr,
        mask_mr,
        vertex_inpaint=cfg.vertex_inpaint,
        method=cfg.inpaint_method,
    )
    return refined_albedo, refined_mr
```

The wrapper’s raw source shows that `inpaint()` delegates to `texture_inpaint(...)` using both a vertex-aware path and a method selector. That is exactly the seam-filling stage you described. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/textureGenPipeline.py))

### `src/export_glb.py`

This file handles steps W and X:

- assign final textures to the renderer
- write a textured OBJ
- convert it to a textured GLB

```python
from __future__ import annotations

from pathlib import Path


def export_textured_glb(pipe, albedo_tex, mr_tex, output_glb: str | Path) -> Path:
    output_glb = Path(output_glb)
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    output_obj = output_glb.with_suffix(".obj")

    pipe.set_texture_albedo(albedo_tex)
    pipe.set_texture_mr(mr_tex)

    generated_glb = Path(pipe.save_mesh(str(output_obj)))

    if generated_glb != output_glb:
        if output_glb.exists():
            output_glb.unlink()
        generated_glb.replace(output_glb)

    return output_glb
```

The wrapper’s save path uses `create_glb_with_pbr_materials(...)`, which is why this route is preferable to a plain `trimesh.export()` when you want a PBR-aware GLB with the expected material textures attached. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/textureGenPipeline.py))

### `src/main.py`

This is the full orchestrator. It wires together every step, saves intermediate outputs, and can be used from a tiny root launcher.

```python
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .bake_texture import bake_textures
from .config import PipelineConfig
from .export_glb import export_textured_glb
from .inpaint_texture import inpaint_textures
from .mesh_generate import export_debug_mesh, generate_raw_mesh, load_shape_pipeline, postprocess_mesh
from .paint_multiview import delight_reference, generate_texture_views, upscale_views
from .preprocess import preprocess_views
from .render_multiview import prepare_uv_and_renderer, render_geometry_maps

LOGGER = logging.getLogger("hy3d-mv")


def make_stage_dirs(output_glb: Path) -> dict[str, Path]:
    run_dir = output_glb.parent / output_glb.stem
    dirs = {
        "run": run_dir,
        "pre": run_dir / "01_preprocessed",
        "mesh": run_dir / "02_mesh",
        "maps_normal": run_dir / "03_normal_maps",
        "maps_position": run_dir / "04_position_maps",
        "reference": run_dir / "05_reference",
        "mv_albedo": run_dir / "06_mv_albedo",
        "mv_mr": run_dir / "07_mv_mr",
        "bake": run_dir / "08_bake",
        "refine": run_dir / "09_refine",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_pils(images: Iterable[Image.Image], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(out_dir / f"{prefix}_{i:02d}.png")


def to_pil_rgb(tex) -> Image.Image:
    if isinstance(tex, Image.Image):
        return tex.convert("RGB")

    if torch.is_tensor(tex):
        t = tex.detach().float().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        arr = t.numpy()
    else:
        arr = np.asarray(tex)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def save_texture_and_mask(texture, mask, out_tex_path: Path, out_mask_path: Path) -> None:
    to_pil_rgb(texture).save(out_tex_path)

    if torch.is_tensor(mask):
        m = mask.detach().float().cpu()
        if m.ndim == 4:
            m = m[0]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        arr = (torch.clamp(m, 0.0, 1.0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_mask_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--back", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shape-subfolder", default="hunyuan3d-dit-v2-mv")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    cfg = PipelineConfig(
        seed=args.seed,
        shape_model_subfolder=args.shape_subfolder,
    )

    output_glb = Path(args.output).resolve()
    stage_dirs = make_stage_dirs(output_glb)

    LOGGER.info("device=%s dtype=%s", cfg.device, cfg.dtype)
    LOGGER.info("shape model=%s / %s", cfg.shape_model_id, cfg.shape_model_subfolder)

    torch.manual_seed(cfg.seed)

    LOGGER.info("loading and preprocessing input views")
    mv_inputs = preprocess_views(
        front_path=args.front,
        left_path=args.left,
        right_path=args.right,
        back_path=args.back,
        cfg=cfg,
    )

    for name, img in mv_inputs.items():
        img.save(stage_dirs["pre"] / f"{name}.png")
        img.getchannel("A").save(stage_dirs["pre"] / f"{name}_alpha.png")

    LOGGER.info("loading multiview shape pipeline")
    shape_pipe = load_shape_pipeline(cfg)

    LOGGER.info("generating raw mesh")
    with torch.inference_mode():
        raw_mesh = generate_raw_mesh(shape_pipe, mv_inputs, cfg)

    LOGGER.info("postprocessing mesh")
    clean_mesh = postprocess_mesh(raw_mesh)
    export_debug_mesh(clean_mesh, stage_dirs["mesh"] / "mesh_untextured.obj")
    export_debug_mesh(clean_mesh, stage_dirs["mesh"] / "mesh_untextured.glb")

    LOGGER.info("uv unwrap and renderer setup")
    paint_pipe, uv_mesh = prepare_uv_and_renderer(clean_mesh, cfg)
    export_debug_mesh(uv_mesh, stage_dirs["mesh"] / "mesh_uv.obj")

    LOGGER.info("rendering normal and position maps")
    with torch.inference_mode():
        normal_maps, position_maps = render_geometry_maps(paint_pipe, cfg)

    save_pils(normal_maps, stage_dirs["maps_normal"], "normal")
    save_pils(position_maps, stage_dirs["maps_position"], "position")

    LOGGER.info("building delighted appearance reference from front view")
    with torch.inference_mode():
        ref_img = delight_reference(mv_inputs["front"], cfg)
    ref_img.save(stage_dirs["reference"] / "reference_delighted.png")

    LOGGER.info("running multiview texture diffusion")
    with torch.inference_mode():
        albedo_views, mr_views = generate_texture_views(
            paint_pipe,
            ref_img,
            normal_maps,
            position_maps,
            cfg,
        )

    save_pils(albedo_views, stage_dirs["mv_albedo"], "albedo")
    save_pils(mr_views, stage_dirs["mv_mr"], "mr")

    LOGGER.info("upscaling generated views to texture target size")
    albedo_views_up = upscale_views(albedo_views, cfg.texture_size)
    mr_views_up = upscale_views(mr_views, cfg.texture_size)

    LOGGER.info("baking multiview images into uv textures")
    with torch.inference_mode():
        tex_albedo, mask_albedo, tex_mr, mask_mr = bake_textures(
            paint_pipe,
            albedo_views_up,
            mr_views_up,
            cfg,
        )

    save_texture_and_mask(
        tex_albedo,
        mask_albedo,
        stage_dirs["bake"] / "baked_albedo.png",
        stage_dirs["bake"] / "baked_albedo_mask.png",
    )
    save_texture_and_mask(
        tex_mr,
        mask_mr,
        stage_dirs["bake"] / "baked_mr.png",
        stage_dirs["bake"] / "baked_mr_mask.png",
    )

    LOGGER.info("refining baked textures with mesh-aware + cv2 inpainting")
    with torch.inference_mode():
        refined_albedo, refined_mr = inpaint_textures(
            paint_pipe,
            tex_albedo,
            mask_albedo,
            tex_mr,
            mask_mr,
            cfg,
        )

    to_pil_rgb(refined_albedo).save(stage_dirs["refine"] / "refined_albedo.png")
    to_pil_rgb(refined_mr).save(stage_dirs["refine"] / "refined_mr.png")

    LOGGER.info("exporting final textured glb")
    final_glb = export_textured_glb(
        paint_pipe,
        refined_albedo,
        refined_mr,
        output_glb,
    )

    LOGGER.info("done: %s", final_glb)
    return 0
```

### Root-level `main.py`

Put this in the project root so you can run the exact command line you requested.

```python
from src.main import main

if __name__ == "__main__":
    raise SystemExit(main())
```

### Run command

```bash
python main.py \
  --front inputs/front.png \
  --left inputs/left.png \
  --right inputs/right.png \
  --back inputs/back.png \
  --output outputs/model.glb
```

### What each file does

`config.py` defines every workflow and path parameter.  
`preprocess.py` normalizes the four views to RGBA 518×518 inputs.  
`mesh_generate.py` runs the 2mv shape pipeline and cleans the geometry.  
`render_multiview.py` UV-unwraps the mesh and renders the six geometry-conditioned maps.  
`paint_multiview.py` builds the delighted appearance anchor and runs multiview texture diffusion.  
`bake_texture.py` projects the generated views into UV space.  
`inpaint_texture.py` fills missing texels.  
`export_glb.py` writes the PBR textures back to the mesh and exports the GLB.  
`src/main.py` orchestrates the whole pipeline and saves your debug artifacts.

## Camera and texture details

### Why six render views if shape starts from four images

Your geometry stage starts from **four external conditioning views**:

- front
- left
- right
- back

But the texture stage renders **six internal geometry views**:

- azimuths: `0, 90, 180, 270, 0, 180`
- elevations: `0, 0, 0, 0, 90, -90`

That means the texture system uses the four cardinal side views plus **top** and **bottom** views to project texture onto surfaces the original four inputs do not directly observe. That is also why the top/bottom views get low bake weights in your workflow: they are coverage helpers, not the primary appearance anchors. The wrapper config and bake calls use the exact same azimuth / elevation / weight arrays as the source of truth for rendering and baking. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/textureGenPipeline.py))

### Why front-view anchoring matters

The 2.1 wrapper’s multiview diffusion net accepts an image list, but in raw source it computes DINO features from `input_images[0]` and passes `input_images[0:1]` into the actual paint pipeline. In other words, **the first reference image is special**. That is the concrete reason to use the **front** view as the single delighted appearance anchor in this standalone project. The other five views for the texture stage come from rendered geometry maps, not from extra appearance-reference images. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/utils/multiview_utils.py))

### How view ordering must stay consistent

The wrapper’s multiview code constructs the condition tensor list like this:

- first half = all normal maps
- second half = all position maps

and it derives `num_view = len(control_images) // 2`. So the view order must stay consistent through:

1. geometry rendering
2. texture-view generation
3. bake projection

If you save the views and later reload them in a different order, the bake will misproject. This is one of the easiest ways to get a texture that “looks almost right” but is shifted or mirrored in UV space. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/utils/multiview_utils.py))

### What the texture stages actually are

The pipeline contains several different image domains. Keeping them distinct makes debugging much easier.

**Source input views** are your original four photographs or renders. They drive the geometry stage.

**Delighted reference image** is the single appearance anchor used for the texture model. Its job is to reduce directional lighting bias so the paint model is not forced to “bake in” strong highlights and shadows.

**Rendered geometry maps** are the six normal maps and six position maps produced from the UV-unwrapped mesh. These are geometric control signals.

**Generated multiview texture images** are the six appearance predictions the diffusion model creates from the delighted reference plus the geometry controls.

**Baked UV texture** is the atlas-space projection of those six generated views onto the mesh UVs.

**Inpainted / refined texture** is the bake result after missing texels and seams have been repaired.

That ordering matches the official 2.0 system description: first create geometry, then synthesize texture for that geometry. UV unwrap must happen before baking because there is no surface-to-atlas projection until UV coordinates exist. Inpainting is needed after baking because some texels remain uncovered or conflicting even when six views are used. ([huggingface.co](https://huggingface.co/tencent/Hunyuan3D-2))

## Validation, optimization, and troubleshooting

### What to inspect after each stage

Inspect the alpha masks first. If the subject is clipped or the mask leaks into the background, the geometry stage is usually worse before it is even asked to solve shape.

Inspect the untextured debug mesh before you touch texturing. If the silhouette, limb thickness, or symmetry is already wrong here, no amount of paint-stage tuning will fix it.

Inspect the UV-unwrapped mesh in a DCC viewer if you suspect seams or missing islands.

Inspect the rendered normal and position maps. They should be smooth, consistent across neighboring views, and should clearly correspond to the same six camera order you will use for baking.

Inspect the generated multiview albedo images. If one view is consistently “off,” check whether your camera ordering got scrambled or whether the delighted reference is too dark / too contrasty.

Inspect the baked mask. Large black or empty regions mean your bake coverage is low, your UV unwrap is poor, your view order is wrong, or your texture views were resized inconsistently before baking.

Inspect the final GLB in a real viewer. If the colors look correct but the mesh appears untextured, the export path probably wrote the textures but failed to attach the material correctly.

### Performance knobs

The official 2.1 repo says the public models need about **10 GB VRAM for shape**, **21 GB for texture**, and **29 GB total** if you do both together on one GPU. The wrapper-side multiview diffusion code additionally enables **model CPU offload**, **VAE slicing**, and **VAE tiling**, which are the first optimizations to keep when memory is tight. ([github.com](https://github.com/tencent-hunyuan/hunyuan3d-2.1))

Practical speed / memory knobs, in descending order of impact:

Reduce `texture_size` from `2048` to `1024`. This is the cheapest memory win.

Reduce `render_size` from `1024` to `768` if bake quality still looks acceptable.

Use `hunyuan3d-dit-v2-mv-fast` or `hunyuan3d-dit-v2-mv-turbo` instead of the standard 2mv checkpoint when geometry speed matters more than fidelity. Those variants are published directly in the 2mv repo. ([huggingface.co](https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main))

Skip Delight if your inputs are already studio-lit or synthetic.

Skip texture generation entirely during mesh debugging and export only the untextured GLB.

Use the older 2.0 fast multiview texturing path if you just need a quicker RGB texture pass rather than the full 2.1 PBR stack. The public project added a fast multiview texture example for that route. ([github.com](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/examples/fast_texture_gen_multiview.py))

### Common errors and fixes

**CUDA mismatch**

If you installed a different torch CUDA wheel from the officially tested stack, rebuild with `torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1`, and the `cu124` wheel. Do not assume newer wheel == better. The public 2.1 instructions test against `cu124`, and issue reports show users running into trouble after installing other wheel variants such as `cu128`. ([github.com](https://github.com/tencent-hunyuan/hunyuan3d-2.1))

**`No module named custom_rasterizer`**

Build the editable extension in the exact `hy3dpaint/custom_rasterizer` tree you import from:

```bash
cd third_party/ComfyUI-Hunyuan3d-2-1/hy3dpaint/custom_rasterizer
pip install -e .
```

The setup file builds a PyTorch CUDA extension named `custom_rasterizer_kernel`. ([raw.githubusercontent.com](https://raw.githubusercontent.com/tencent-hunyuan/hunyuan3d-2.1/main/hy3dpaint/custom_rasterizer/setup.py))

**`libcudart.so` not found**

Set `CUDA_HOME` and add `$CUDA_HOME/lib64` to `LD_LIBRARY_PATH` before rebuilding. PyTorch’s extension docs explicitly note that CUDA extensions link against `cudart`. ([docs.pytorch.org](https://docs.pytorch.org/docs/2.11/cpp_extension.html))

**`no kernel image available for execution on the device`**

Set `TORCH_CUDA_ARCH_LIST` to match your GPU, then reinstall the extension. This is exactly what PyTorch recommends for extension arch targeting. ([docs.pytorch.org](https://docs.pytorch.org/docs/2.11/cpp_extension.html))

**Torch extension compile failures**

Make sure `ninja-build` is installed, because PyTorch’s extension build system uses the Ninja backend by default when available. Also verify `nvcc --version`, `CUDA_HOME`, and your torch CUDA wheel before building. ([docs.pytorch.org](https://docs.pytorch.org/docs/2.11/cpp_extension.html))

**Missing `pygltflib`**

Install it explicitly:

```bash
pip install pygltflib
```

The community wrapper requirements include `pygltflib`; if it is missing, GLB assembly helpers can fail. ([github.com](https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1/blob/main/requirements.txt))

**Missing `xatlas` or UV unwrap dependency**

Install:

```bash
pip install xatlas==0.0.10
```

The official 2.1 requirements pin `xatlas==0.0.10`, and the wrapper’s texturing code calls `mesh_uv_wrap(...)` before rendering / baking. ([github.com](https://github.com/tencent-hunyuan/hunyuan3d-2.1))

**GLB exports but appears untextured**

Use the wrapper’s `set_texture_albedo(...)`, `set_texture_mr(...)`, and `save_mesh(...)` path instead of a plain `trimesh.export()`. The wrapper explicitly routes the export through a GLB material helper. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/textureGenPipeline.py))

**Texture is misaligned with the mesh**

Most often one of three causes:

- you changed the six-view order between render and bake
- you baked before UV unwrap
- you resized only some view images before the bake

Keep the same azimuth/elevation order for render, generation, and bake, and unwrap before you render your control maps. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/textureGenPipeline.py))

**Wrong camera order**

Do not infer order from filenames alone. Use the arrays in `config.py` as the single source of truth, and save the view index with each generated map and texture view. The multiview code assumes the condition order is stable. ([raw.githubusercontent.com](https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/main/hy3dpaint/utils/multiview_utils.py))

**Bad alpha or background removal**

If your input already contains a good alpha channel, preserve it. Only run background removal when the alpha is effectively solid / absent. That matches the official example pattern and avoids destroying already-good cutouts. ([github.com](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/examples/fast_texture_gen_multiview.py))

**RunPod file/cache issues**

Put `HF_HOME` under `/workspace` or on a network volume so the heavy model downloads survive restarts. Runpod’s storage docs explain the persistence behavior of volume disk and network volumes. ([docs.runpod.io](https://docs.runpod.io/pods/storage/types))

**RunPod port confusion**

If you expose a web process for debugging, bind it to `0.0.0.0` and expose that HTTP port in the Pod config. That is how Runpod’s proxying expects the service to be published. ([docs.runpod.io](https://docs.runpod.io/pods/configuration/expose-ports))