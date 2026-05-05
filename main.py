"""
Hunyuan3D Multiview Pipeline — CLI entry point.

Wires all pipeline stages into a single, resumable command.

Stage order:
  remove_background  — rembg + center + resize → clean RGBA
  canonical_multiview— MV-Adapter i2mv SDXL → front/right/back/left (opt-in)
  preprocess         — white/gray compositing for shape/paint models
  mesh_generate      — Hunyuan3D DiT → raw mesh
  mesh_postprocess   — cleanup + decimation
  render_multiview   — UV unwrap + conditioning maps
  paint_multiview    — HunyuanPaint MR; canonical views as albedo when active
  bake_texture       — cosine-weighted UV back-projection
  inpaint_texture    — two-pass UV hole filling
  export_glb         — PBR GLB export

Usage:
  # Single-image mode (Path A — v2.1 single-image DiT)
  python main.py \\
      --image inputs/front.png \\
      --output outputs/model.glb \\
      --shape-steps 50 \\
      --paint-steps 10 \\
      --seed 42

  # Four-view mode (Path B — v2.0 2mv DiT)
  python main.py \\
      --front inputs/front.png --left inputs/left.png \\
      --right inputs/right.png --back inputs/back.png \\
      --output outputs/model.glb \\
      --use-multiview-shape \\
      --shape-steps 30

  # Canonical multiview (MV-Adapter generates consistent 4-view dataset)
  python main.py \\
      --image inputs/front.png \\
      --output outputs/model.glb \\
      --canonical-multiview \\
      --canonical-use-depth

  # Mesh-only (skip texturing)
  python main.py --image inputs/front.png --output outputs/model.glb --no-texture

  # Full pipeline with all intermediate outputs saved
  python main.py --image inputs/front.png --output outputs/model.glb --save-intermediates

Dual-platform note:
  - macOS: All imports succeed; GPU stages (shape gen, paint, render) raise
    ImportError or RuntimeError at runtime because CUDA extensions are absent.
  - RunPod: Full end-to-end execution.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hunyuan3D Multiview Pipeline — image(s) to PBR GLB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Single image → GLB
  python main.py --image inputs/object.png --output outputs/object.glb

  # Four-view folder → GLB (multiview DiT)
  python main.py \\
      --front inputs/obj-front.png --left inputs/obj-left.png \\
      --right inputs/obj-right.png --back inputs/obj-back.png \\
      --output outputs/object.glb --use-multiview-shape

  # Mesh-only (no texturing)
  python main.py --image inputs/object.png --output outputs/object.glb --no-texture

  # Save all intermediates for debugging
  python main.py --image inputs/object.png --output outputs/object.glb --save-intermediates
""",
    )

    # ---- Input images (mutually exclusive groups) ---------------------------
    single_group = parser.add_argument_group("single-image input (Path A)")
    single_group.add_argument(
        "--image", type=Path, default=None, metavar="FILE",
        help="Front-view image for single-image shape generation (v2.1 DiT).",
    )

    mv_group = parser.add_argument_group("four-view input (Path B)")
    mv_group.add_argument(
        "--front", type=Path, default=None, metavar="FILE",
        help="Front-view image for four-view shape generation.",
    )
    mv_group.add_argument(
        "--left", type=Path, default=None, metavar="FILE",
        help="Left-view image.",
    )
    mv_group.add_argument(
        "--right", type=Path, default=None, metavar="FILE",
        help="Right-view image.",
    )
    mv_group.add_argument(
        "--back", type=Path, default=None, metavar="FILE",
        help="Back-view image.",
    )

    # ---- Output / mode ------------------------------------------------------
    parser.add_argument(
        "--output", "-o", type=Path, required=True, metavar="FILE",
        help="Output GLB file path (e.g. outputs/model.glb).",
    )
    parser.add_argument(
        "--no-texture", action="store_true",
        help="Skip texturing — export untextured GLB after mesh postprocessing.",
    )
    parser.add_argument(
        "--save-intermediates", action="store_true",
        help="Save all intermediate outputs (meshes, maps, textures) alongside final GLB.",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep intermediate OBJ and JPEG texture files after GLB export.",
    )

    # ---- Shape parameters ---------------------------------------------------
    shape_group = parser.add_argument_group("shape generation parameters")
    shape_group.add_argument(
        "--shape-steps", type=int, default=50, metavar="N",
        help="Shape diffusion steps (default: 50).",
    )
    shape_group.add_argument(
        "--guidance-scale", type=float, default=5.0, metavar="SCALE",
        help="Shape guidance scale (default: 5.0).",
    )
    shape_group.add_argument(
        "--octree-resolution", type=int, default=384, metavar="RES",
        help="Marching-cubes octree resolution (default: 384; higher = more detail).",
    )
    shape_group.add_argument(
        "--use-multiview-shape", action="store_true",
        help="Use the v2.0 four-view DiT for shape generation (requires --front/left/right/back).",
    )
    shape_group.add_argument(
        "--target-faces", type=int, default=200_000, metavar="N",
        help="Target face count after mesh decimation (default: 200000).",
    )

    # ---- Paint parameters ---------------------------------------------------
    paint_group = parser.add_argument_group("texture paint parameters")
    paint_group.add_argument(
        "--paint-steps", type=int, default=10, metavar="N",
        help="Paint diffusion steps (default: 10).",
    )
    paint_group.add_argument(
        "--paint-guidance", type=float, default=3.0, metavar="SCALE",
        help="Paint guidance scale (default: 3.0).",
    )
    paint_group.add_argument(
        "--texture-size", type=int, default=4096, metavar="PX",
        help="Final UV texture atlas resolution (default: 4096).",
    )
    paint_group.add_argument(
        "--view-size", type=int, default=512, metavar="PX",
        help="Diffusion model view resolution (default: 512).",
    )
    paint_group.add_argument(
        "--render-size", type=int, default=None, metavar="PX",
        help=(
            "Conditioning map / bake render resolution (default: max(2048, texture_size)). "
            "Must be >= texture_size for artifact-free baking."
        ),
    )
    paint_group.add_argument(
        "--no-delight", action="store_true",
        help="Disable the optional Light_Shadow_Remover delight step.",
    )
    paint_group.add_argument(
        "--use-realesrgan", action="store_true",
        help="Use RealESRGAN 4× for view upscaling instead of Lanczos (slower, higher quality).",
    )
    paint_group.add_argument(
        "--inpaint-method", choices=["NS", "TELEA"], default="NS",
        help="OpenCV inpainting method for uncovered UV texels (default: NS).",
    )
    paint_group.add_argument(
        "--no-vertex-inpaint", action="store_true",
        help="Disable vertex-aware inpainting (pass 1); use cv2 only.",
    )

    # ---- Canonical multiview (MV-Adapter) -----------------------------------
    canon_group = parser.add_argument_group(
        "canonical multiview parameters (--canonical-multiview)"
    )
    canon_group.add_argument(
        "--canonical-multiview", action="store_true",
        help=(
            "Run the canonical_multiview stage (MV-Adapter i2mv SDXL) before "
            "preprocess. Generates consistent front/right/back/left views from "
            "the anchor image. Requires ~14 GB VRAM."
        ),
    )
    canon_group.add_argument(
        "--canonical-steps", type=int, default=50, metavar="N",
        help="MV-Adapter diffusion steps (default: 50).",
    )
    canon_group.add_argument(
        "--canonical-guidance", type=float, default=3.0, metavar="SCALE",
        help="MV-Adapter guidance scale (default: 3.0).",
    )
    canon_group.add_argument(
        "--canonical-ref-scale", type=float, default=1.0, metavar="SCALE",
        help="MV-Adapter reference conditioning scale (default: 1.0).",
    )
    canon_group.add_argument(
        "--canonical-use-depth", action="store_true",
        help="Add depth ControlNet conditioning (DPT-hybrid-midas).",
    )
    canon_group.add_argument(
        "--canonical-depth-scale", type=float, default=0.5, metavar="SCALE",
        help="Depth ControlNet scale (default: 0.5).",
    )
    canon_group.add_argument(
        "--canonical-use-canny", action="store_true",
        help="Add weak canny edge ControlNet conditioning.",
    )
    canon_group.add_argument(
        "--canonical-canny-scale", type=float, default=0.2, metavar="SCALE",
        help="Canny ControlNet scale (default: 0.2).",
    )
    canon_group.add_argument(
        "--canonical-prompt", type=str, default="high quality", metavar="TEXT",
        help="Generation prompt for MV-Adapter (default: 'high quality').",
    )

    # ---- Reproducibility / misc ---------------------------------------------
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--remove-bg", action="store_true",
        help="Run rembg background removal on input images before processing.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", metavar="DEVICE",
        help="PyTorch device (default: cuda).",
    )

    return parser


# ---------------------------------------------------------------------------
# Intermediate-output helpers
# ---------------------------------------------------------------------------

def _stage_dir(intermediates_root: Path, n: int, name: str) -> Path:
    """Return a stage sub-directory, creating it if intermediates are enabled."""
    d = intermediates_root / f"{n:02d}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_images(images, directory: Path, prefix: str) -> None:
    """Save a list of PIL images as numbered PNGs."""
    for i, img in enumerate(images):
        img.save(str(directory / f"{prefix}_{i:02d}.png"))


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _run_remove_background(args, cfg, intermediates_root: Path | None):
    """Stage 1 — Load images, run background removal, center, resize → clean RGBA."""
    from src.preprocess import collect_views, prepare_rgba_views, load_image_rgba

    print("\n[remove_background] Loading images and removing backgrounds ...")

    if args.image:
        logger.info("Mode: single-image  path=%s", args.image)
        raw = {"front": load_image_rgba(args.image)}
        cfg.use_multiview_shape = False
    elif args.front:
        logger.info("Mode: four-view  front=%s", args.front)
        raw = collect_views(
            front=str(args.front),
            left=str(args.left) if args.left else None,
            right=str(args.right) if args.right else None,
            back=str(args.back) if args.back else None,
        )
        cfg.use_multiview_shape = True
    else:
        raise ValueError(
            "Provide --image for single-image mode or --front for four-view mode."
        )

    rgba_views = prepare_rgba_views(raw, remove_bg=args.remove_bg)

    if intermediates_root is not None:
        d = _stage_dir(intermediates_root, 1, "remove_background")
        for name, img in rgba_views.items():
            img.save(str(d / f"{name}.png"))

    return rgba_views


def _run_canonical_multiview(args, rgba_views, cfg, intermediates_root: Path | None):
    """Stage 2 (optional) — MV-Adapter i2mv SDXL → canonical 4 views."""
    if not args.canonical_multiview:
        return None

    from src.canonical_multiview import generate_canonical_views
    from src.config import CameraConfig

    print("\n[canonical_multiview] Generating canonical front/right/back/left views ...")

    # Apply CLI args to cfg.canonical
    cfg.canonical.steps                        = args.canonical_steps
    cfg.canonical.guidance_scale               = args.canonical_guidance
    cfg.canonical.reference_conditioning_scale = args.canonical_ref_scale
    cfg.canonical.use_depth                    = args.canonical_use_depth
    cfg.canonical.depth_scale                  = args.canonical_depth_scale
    cfg.canonical.use_canny                    = args.canonical_use_canny
    cfg.canonical.canny_scale                  = args.canonical_canny_scale
    cfg.canonical.prompt                       = args.canonical_prompt

    save_dir = _stage_dir(intermediates_root, 2, "canonical_multiview") \
               if intermediates_root else None

    front_rgba = rgba_views["front"]
    canonical_views = generate_canonical_views(front_rgba, cfg.canonical, save_dir=save_dir)

    # Switch to 4-camera layout for downstream bake
    cfg.camera = CameraConfig.four_cardinal()
    cfg.use_multiview_shape = True

    logger.info("Canonical views: %s", list(canonical_views.keys()))
    return canonical_views


def _run_preprocess(rgba_views, cfg, intermediates_root: Path | None, stage_num: int = 3):
    """Stage 3 — Compositing only: white (shape) + gray (paint) from clean RGBA."""
    from src.preprocess import composite_views

    print("\n[preprocess] Compositing views for shape/paint models ...")

    processed = composite_views(rgba_views)
    shape_views = {k: v["shape"] for k, v in processed.items()}

    if intermediates_root is not None:
        d = _stage_dir(intermediates_root, stage_num, "preprocess")
        for name, entry in processed.items():
            entry["shape"].save(str(d / f"shape_{name}.png"))
            entry["paint"].save(str(d / f"paint_{name}.png"))

    return shape_views


def _run_mesh_generate(args, shape_views, cfg, intermediates_root: Path | None):
    """Steps C + D: shape pipeline load + mesh generation."""
    from src.mesh_generate import load_shape_pipeline_auto, generate_mesh

    print(
        f"\n[shape] Generating mesh  steps={cfg.shape_steps}  "
        f"guidance={cfg.shape_guidance_scale}  res={cfg.octree_resolution} ..."
    )

    t0 = time.perf_counter()
    pipeline = load_shape_pipeline_auto(cfg)
    raw_mesh = generate_mesh(pipeline, shape_views, cfg)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Mesh generated in %.1f s — %d vertices, %d faces.",
        elapsed, len(raw_mesh.vertices), len(raw_mesh.faces),
    )

    if intermediates_root is not None:
        d = _stage_dir(intermediates_root, 2, "mesh")
        raw_mesh.export(str(d / "mesh_raw.glb"))
        logger.info("Saved mesh_raw.glb to %s.", d)

    return raw_mesh


def _run_postprocess(raw_mesh, cfg, intermediates_root: Path | None):
    """Step E: mesh cleanup and normalization."""
    from src.mesh_postprocess import postprocess_mesh, save_mesh

    print(
        f"\n[mesh] Postprocessing  target_faces={cfg.target_faces} "
        f"normalize={cfg.normalize_mesh} ..."
    )

    post_mesh = postprocess_mesh(
        raw_mesh, target_faces=cfg.target_faces, normalize=cfg.normalize_mesh
    )
    logger.info(
        "Postprocess done: %d vertices, %d faces.",
        len(post_mesh.vertices), len(post_mesh.faces),
    )

    if intermediates_root is not None:
        d = intermediates_root / "02_mesh"
        save_mesh(post_mesh, d / "mesh_postprocessed.glb")
        logger.info("Saved mesh_postprocessed.glb to %s.", d)

    return post_mesh


def _run_uv_and_render(post_mesh, cfg, intermediates_root: Path | None):
    """Steps F + H + I: UV unwrap, renderer init, conditioning maps."""
    from src.render_multiview import uv_unwrap_mesh, PaintPipeline, render_conditioning_maps

    print("\n[render] UV unwrapping mesh ...")
    uv_mesh = uv_unwrap_mesh(post_mesh)
    logger.info(
        "UV unwrap done: %d vertices, %d faces.",
        len(uv_mesh.vertices), len(uv_mesh.faces),
    )

    if intermediates_root is not None:
        d = intermediates_root / "02_mesh"
        d.mkdir(parents=True, exist_ok=True)
        uv_mesh.export(str(d / "mesh_uv.glb"))

    print("\n[render] Initializing paint renderer ...")
    paint_pipeline = PaintPipeline(cfg)
    paint_pipeline.load_mesh(uv_mesh)

    print(f"\n[render] Rendering conditioning maps ({len(cfg.camera.azimuths)} views) ...")
    normal_maps, position_maps = render_conditioning_maps(paint_pipeline, cfg)

    if intermediates_root is not None:
        _save_images(normal_maps,   _stage_dir(intermediates_root, 3, "normal_maps"),   "normal")
        _save_images(position_maps, _stage_dir(intermediates_root, 4, "position_maps"), "position")

    return uv_mesh, paint_pipeline, normal_maps, position_maps


def _run_paint(
    front_rgba,
    normal_maps,
    position_maps,
    cfg,
    intermediates_root: Path | None,
    *,
    canonical_views=None,
):
    """Steps J + K + L: delight, multiview diffusion, upscale.

    When ``canonical_views`` is provided the canonical RGB views are used as
    albedo and HunyuanPaint contributes only the MR channel.
    """
    from src.paint_multiview import MultiviewDiffusionNet, delight_reference, upscale_views

    print("\n[paint] Delight reference image ...")
    reference = delight_reference(front_rgba, cfg)

    if intermediates_root is not None:
        d = _stage_dir(intermediates_root, 5, "reference")
        reference.save(str(d / "reference_delighted.png"))

    print(f"\n[paint] Loading paint model ...")
    mvd = MultiviewDiffusionNet(cfg)

    print(
        f"\n[paint] Multiview diffusion  steps={cfg.paint_steps}  "
        f"guidance={cfg.paint_guidance_scale} ..."
    )
    t0 = time.perf_counter()
    paint_out = mvd(reference, normal_maps, position_maps, cfg)
    elapsed = time.perf_counter() - t0
    mr_views = paint_out["mr"]
    logger.info("Diffusion done in %.1f s — %d MR views.", elapsed, len(mr_views))

    if intermediates_root is not None:
        if canonical_views is None:
            albedo_views = paint_out["albedo"]
            _save_images(albedo_views, _stage_dir(intermediates_root, 6, "mv_albedo"), "albedo")
        _save_images(mr_views, _stage_dir(intermediates_root, 7, "mv_mr"), "mr")

    print(f"\n[paint] Upscaling views to {cfg.render_size} px ...")
    ckpt = cfg.models_dir / "RealESRGAN_x4plus.pth" if cfg.use_realesrgan else None

    if canonical_views is not None:
        # Use canonical views as albedo; HunyuanPaint only provides MR.
        canon_order = ["front", "right", "back", "left"]
        albedo_up = upscale_views(
            [canonical_views[k].convert("RGB") for k in canon_order],
            cfg.render_size,
            use_realesrgan=cfg.use_realesrgan,
            realesrgan_ckpt=ckpt,
        )
        mr_up = upscale_views(
            mr_views[:4],
            cfg.render_size,
            use_realesrgan=cfg.use_realesrgan,
            realesrgan_ckpt=ckpt,
        )
        logger.info("Albedo source: canonical_views (%d views).", len(albedo_up))
    else:
        albedo_views = paint_out["albedo"]
        albedo_up = upscale_views(
            albedo_views,
            cfg.render_size,
            use_realesrgan=cfg.use_realesrgan,
            realesrgan_ckpt=ckpt,
        )
        mr_up = upscale_views(
            mr_views,
            cfg.render_size,
            use_realesrgan=cfg.use_realesrgan,
            realesrgan_ckpt=ckpt,
        )
        logger.info("Albedo source: hunyuan_paint (%d views).", len(albedo_up))

    mvd.unload()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return albedo_up, mr_up


def _run_bake(paint_pipeline, albedo_up, mr_up, cfg, intermediates_root: Path | None):
    """Step M: cosine-weighted UV back-projection bake."""
    from src.bake_texture import bake_multiview_textures

    print("\n[bake] Baking multiview views into UV texture maps ...")
    save_dir = _stage_dir(intermediates_root, 8, "bake") if intermediates_root else None

    return bake_multiview_textures(
        paint_pipeline, albedo_up, mr_up, cfg, save_dir=save_dir
    )


def _run_inpaint(
    paint_pipeline,
    texture_albedo, mask_albedo,
    texture_mr, mask_mr,
    cfg,
    intermediates_root: Path | None,
):
    """Step N: two-pass UV texture inpainting."""
    from src.inpaint_texture import inpaint_textures

    print(
        f"\n[inpaint] Filling uncovered texels  "
        f"vertex={cfg.vertex_inpaint}  method={cfg.inpaint_method} ..."
    )
    save_dir = _stage_dir(intermediates_root, 9, "refine") if intermediates_root else None

    return inpaint_textures(
        paint_pipeline,
        texture_albedo, mask_albedo,
        texture_mr, mask_mr,
        vertex_inpaint=cfg.vertex_inpaint,
        method=cfg.inpaint_method,
        save_dir=save_dir,
    )


def _run_export(
    paint_pipeline,
    refined_albedo,
    refined_mr,
    uv_mesh,
    output_path: Path,
    cleanup_obj: bool,
):
    """Step O: PBR GLB export."""
    from src.export_glb import export_textured_mesh

    print(f"\n[export] Exporting PBR GLB to {output_path} ...")
    return export_textured_mesh(
        paint_pipeline,
        refined_albedo, refined_mr,
        uv_mesh,
        output_path,
        cleanup_obj=cleanup_obj,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Validate: at least one input provided
    if not args.image and not args.front:
        parser.error("Provide --image (single-image mode) or --front (four-view mode).")

    # Validate: four-view mode consistency
    if args.use_multiview_shape and not args.front:
        parser.error("--use-multiview-shape requires --front (and optionally --left/--right/--back).")

    # ---- Build config -------------------------------------------------------
    from src.config import PipelineConfig

    cfg = PipelineConfig(device=args.device)
    cfg.shape_steps           = args.shape_steps
    cfg.shape_guidance_scale  = args.guidance_scale
    cfg.octree_resolution     = args.octree_resolution
    cfg.use_multiview_shape   = args.use_multiview_shape
    cfg.target_faces          = args.target_faces
    cfg.paint_steps           = args.paint_steps
    cfg.paint_guidance_scale  = args.paint_guidance
    cfg.texture_size          = args.texture_size
    cfg.view_size             = args.view_size
    cfg.render_size           = args.render_size or max(2048, args.texture_size)
    cfg.use_delight           = not args.no_delight
    cfg.use_realesrgan        = args.use_realesrgan
    cfg.inpaint_method        = args.inpaint_method
    cfg.vertex_inpaint        = not args.no_vertex_inpaint
    cfg.seed                  = args.seed
    # canonical args applied later in _run_canonical_multiview

    # ---- Intermediate outputs root -----------------------------------------
    output_path = Path(args.output).with_suffix(".glb")
    intermediates_root: Path | None = None
    if args.save_intermediates:
        intermediates_root = output_path.parent / output_path.stem
        intermediates_root.mkdir(parents=True, exist_ok=True)
        logger.info("Saving intermediates to %s.", intermediates_root)

    # ---- Apply canonical multiview config -----------------------------------
    cfg.canonical.steps                        = args.canonical_steps
    cfg.canonical.guidance_scale               = args.canonical_guidance
    cfg.canonical.reference_conditioning_scale = args.canonical_ref_scale
    cfg.canonical.use_depth                    = args.canonical_use_depth
    cfg.canonical.depth_scale                  = args.canonical_depth_scale
    cfg.canonical.use_canny                    = args.canonical_use_canny
    cfg.canonical.canny_scale                  = args.canonical_canny_scale
    cfg.canonical.prompt                       = args.canonical_prompt

    # ---- Run pipeline -------------------------------------------------------
    t_start = time.perf_counter()

    try:
        # Stage 1: load + background removal → clean RGBA
        rgba_views = _run_remove_background(args, cfg, intermediates_root)

        # Stage 2 (optional): MV-Adapter canonical views
        canonical_views = _run_canonical_multiview(args, rgba_views, cfg, intermediates_root)

        # Use canonical views downstream if generated
        active_rgba_views = canonical_views if canonical_views is not None else rgba_views

        # Stage 3: white/gray compositing → shape_views
        shape_views = _run_preprocess(active_rgba_views, cfg, intermediates_root, stage_num=3)

        # The front reference for paint is the canonical front (if available) or original
        front_rgba = active_rgba_views["front"]

        raw_mesh   = _run_mesh_generate(args, shape_views, cfg, intermediates_root)
        post_mesh  = _run_postprocess(raw_mesh, cfg, intermediates_root)

        if args.no_texture:
            from src.mesh_postprocess import save_mesh
            save_mesh(post_mesh, output_path.with_suffix(".glb"))
            print(f"\nMesh-only output saved to {output_path.with_suffix('.glb')}")
            return

        uv_mesh, paint_pipeline, normal_maps, position_maps = _run_uv_and_render(
            post_mesh, cfg, intermediates_root
        )
        albedo_up, mr_up = _run_paint(
            front_rgba, normal_maps, position_maps, cfg, intermediates_root,
            canonical_views=canonical_views,
        )
        texture_albedo, mask_albedo, texture_mr, mask_mr = _run_bake(
            paint_pipeline, albedo_up, mr_up, cfg, intermediates_root
        )
        refined_albedo, refined_mr = _run_inpaint(
            paint_pipeline,
            texture_albedo, mask_albedo,
            texture_mr, mask_mr,
            cfg,
            intermediates_root,
        )
        glb_path = _run_export(
            paint_pipeline, refined_albedo, refined_mr, uv_mesh,
            output_path, cleanup_obj=not args.no_cleanup,
        )

        elapsed = time.perf_counter() - t_start
        print(f"\nDone in {elapsed:.1f} s — output: {glb_path}")

    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
