"""
Phase 5 test — bake_texture.py + inpaint_texture.py + export_glb.py

Runs the full Phase 5 pipeline (Steps M–O): UV texture bake, inpaint, and
PBR GLB export.  Accepts a pre-rendered set of upscaled albedo/MR view images
(from a prior Phase 4 run) or regenerates them from scratch by running the
Phase 4 pipeline inline.

Requires a GPU with the custom_rasterizer and mesh_inpaint_processor CUDA
extensions compiled.

Outputs written to  outputs/test/phase5/<timestamp>/:
  baked_albedo.png           — UV texture after baking
  baked_albedo_mask.png      — trust mask (white = covered)
  baked_mr.png               — MR texture after baking
  baked_mr_mask.png          — trust mask for MR
  refined_albedo.png         — albedo after inpainting
  refined_mr.png             — MR after inpainting
  output.glb                 — final PBR GLB
  mesh_uv.glb                — UV-unwrapped mesh (for validation)
  metrics.log / metrics.json — timing + VRAM

Usage:
  # Full pipeline from a single input image
  python scripts/test_phase5.py --image inputs/object.png

  # From a four-view folder
  python scripts/test_phase5.py --multiview inputs/object/

  # Skip mesh generation; use existing postprocessed GLB + Phase 4 outputs
  python scripts/test_phase5.py \\
      --glb    outputs/test/phase4/.../mesh_postprocessed.glb \\
      --albedo outputs/test/phase4/.../albedo_upscaled_*.png \\
      --mr     outputs/test/phase4/.../mr_upscaled_*.png \\
      --ref    inputs/object.png

  --image          Input image — full pipeline from scratch.
  --multiview      Folder of 4 orientation-suffixed images — full multiview pipeline.
  --glb            Pre-generated postprocessed GLB (skips mesh + paint stages).
  --albedo         Glob or space-separated list of upscaled albedo PNGs.
  --mr             Glob or space-separated list of upscaled MR PNGs.
  --ref            Reference front-view image (required when --glb is used
                   without --image, to reload the mesh into the renderer).
  --shape-steps    Shape diffusion steps when generating mesh (default 50).
  --paint-steps    Paint diffusion steps (default 10).
  --skip-bake      Load baked textures from --baked-albedo / --baked-mr paths.
  --baked-albedo   Pre-baked albedo tensor path (skip bake, go straight to inpaint).
  --baked-mr       Pre-baked MR tensor path.
  --seed           Random seed.
  --texture-size   UV atlas resolution (default 4096).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import trimesh  # noqa: E402
from scripts.test_utils import RunLogger, resolve_seed  # noqa: E402

OUTPUT_DIR = (
    _PROJECT_ROOT / "outputs" / "test" / "phase5"
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log = RunLogger(OUTPUT_DIR, phase="Phase 5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_images_from_paths(paths: list[str]) -> list:
    """Load a list of image paths as PIL images."""
    from PIL import Image

    result = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        result.append(img)
    return result


def _save_image(img, name: str) -> None:
    img.save(str(OUTPUT_DIR / name))
    print(f"      saved → {name}")


def _save_images(images, prefix: str) -> None:
    for i, img in enumerate(images):
        img.save(str(OUTPUT_DIR / f"{prefix}_{i:02d}.png"))
    print(f"      saved {len(images)} → {prefix}_*.png")


# ---------------------------------------------------------------------------
# Stage: Build UV mesh + PaintPipeline (shared across bake + export)
# ---------------------------------------------------------------------------

def build_pipeline_and_mesh(args, cfg):
    """
    Build the PaintPipeline and UV-unwrapped mesh required for bake + export.

    Returns (uv_mesh, paint_pipeline, albedo_up, mr_up).
    """
    from src.render_multiview import uv_unwrap_mesh, PaintPipeline, render_conditioning_maps

    # ---- Resolve postprocessed mesh ----------------------------------------
    if args.glb:
        print(f"\n[mesh] Loading postprocessed mesh from {args.glb}")
        loaded = trimesh.load(str(args.glb))
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        post_mesh = loaded
        log.metric("mesh_source", str(args.glb))

    else:
        post_mesh = _generate_mesh(args, cfg)

    # ---- UV unwrap ---------------------------------------------------------
    with log.step("uv_unwrap_mesh"):
        uv_mesh = uv_unwrap_mesh(post_mesh)
        log.metric("uv_vertices", len(uv_mesh.vertices), unit="verts")
        log.metric("uv_faces",    len(uv_mesh.faces),    unit="faces")

    uv_mesh.export(str(OUTPUT_DIR / "mesh_uv.glb"))
    print("      saved → mesh_uv.glb")

    # ---- Paint pipeline (renderer) ----------------------------------------
    with log.step("PaintPipeline init"):
        paint_pipeline = PaintPipeline(cfg)

    with log.step("load_mesh into renderer"):
        paint_pipeline.load_mesh(uv_mesh)

    # ---- Resolve albedo/MR views -------------------------------------------
    if args.glb and args.albedo:
        # Pre-rendered views provided directly
        print("\n[views] Loading pre-rendered upscaled views from --albedo/--mr ...")
        with log.step("load albedo views"):
            albedo_up = _load_images_from_paths(args.albedo)
            log.metric("albedo_views", len(albedo_up), unit="images")
        with log.step("load MR views"):
            mr_up = _load_images_from_paths(args.mr)
            log.metric("mr_views", len(mr_up), unit="images")

    else:
        # Run Phase 4 (render + paint) inline to get views
        albedo_up, mr_up = _run_phase4_paint(args, cfg, uv_mesh, paint_pipeline)

    return uv_mesh, paint_pipeline, albedo_up, mr_up


def _generate_mesh(args, cfg):
    """Run Phase 3 pipeline to produce a postprocessed mesh."""
    from src.preprocess import (
        load_image_rgba, compose_over_white,
        scan_multiview_folder, collect_views, preprocess_all_views,
    )
    from src.mesh_generate import load_shape_pipeline_auto, generate_mesh
    from src.mesh_postprocess import postprocess_mesh, save_mesh

    if args.multiview:
        print(f"\n[mesh] Generating from multiview folder: {args.multiview}")
        with log.step("scan_multiview_folder"):
            view_paths = scan_multiview_folder(args.multiview)
            for orient, p in view_paths.items():
                log.metric(f"input_{orient}", p.name)
        with log.step("load + preprocess all views"):
            raw_views = collect_views(**{k: str(v) for k, v in view_paths.items()})
            processed = preprocess_all_views(raw_views, remove_bg=False)
            shape_views = {k: v["shape"] for k, v in processed.items()}
        cfg.use_multiview_shape = True

    else:
        print(f"\n[mesh] Generating from {args.image}")
        with log.step("load_image + white composite"):
            raw_img = load_image_rgba(args.image)
            shape_views = {"front": compose_over_white(raw_img)}
            log.metric(
                "input_image",
                f"{args.image.name} {raw_img.size[0]}×{raw_img.size[1]}",
                unit="px",
            )
        cfg.use_multiview_shape = False

    with log.step("load_shape_pipeline_auto"):
        pipeline = load_shape_pipeline_auto(cfg)

    with log.step(f"generate_mesh (steps={cfg.shape_steps})"):
        raw_mesh = generate_mesh(pipeline, shape_views, cfg)
        log.metric("raw_vertices", len(raw_mesh.vertices), unit="verts")
        log.metric("raw_faces",    len(raw_mesh.faces),    unit="faces")

    with log.step(f"postprocess_mesh (target={cfg.target_faces})"):
        post_mesh = postprocess_mesh(
            raw_mesh, target_faces=cfg.target_faces, normalize=cfg.normalize_mesh
        )
        log.metric("post_vertices", len(post_mesh.vertices), unit="verts")
        log.metric("post_faces",    len(post_mesh.faces),    unit="faces")

    save_mesh(post_mesh, OUTPUT_DIR / "mesh_postprocessed.glb")
    print("      saved → mesh_postprocessed.glb")
    return post_mesh


def _run_phase4_paint(args, cfg, uv_mesh, paint_pipeline):
    """Run Phase 4 inline (conditioning maps + diffusion + upscale)."""
    import torch
    from src.render_multiview import render_conditioning_maps
    from src.paint_multiview import MultiviewDiffusionNet, delight_reference, upscale_views
    from src.preprocess import load_image_rgba, scan_multiview_folder

    # Resolve front reference
    if args.image:
        front_path = args.image
    elif args.multiview:
        view_paths = scan_multiview_folder(args.multiview)
        front_path = view_paths["front"]
    elif args.ref:
        front_path = args.ref
    else:
        raise ValueError(
            "No front-view reference available. "
            "Provide --image, --multiview, or --ref."
        )

    with log.step(f"render_conditioning_maps ({len(cfg.camera.azimuths)} views)"):
        normal_maps, position_maps = render_conditioning_maps(paint_pipeline, cfg)
        log.metric("normal_maps",   len(normal_maps),   unit="images")
        log.metric("position_maps", len(position_maps), unit="images")

    _save_images(normal_maps,   "normal")
    _save_images(position_maps, "position")

    paint_model_dir = cfg.models_dir / "hunyuan3d-paintpbr-v2-1"
    if not paint_model_dir.exists():
        raise FileNotFoundError(
            f"Paint model not found at {paint_model_dir}.\n"
            "Run:  python scripts/download_models.py"
        )

    with log.step("delight_reference"):
        front_rgba = load_image_rgba(front_path)
        reference = delight_reference(front_rgba, cfg)
        _save_image(reference, "reference_delighted.png")

    with log.step("MultiviewDiffusionNet init"):
        mvd = MultiviewDiffusionNet(cfg)

    with log.step(f"paint diffusion (steps={cfg.paint_steps})"):
        paint_out = mvd(reference, normal_maps, position_maps, cfg)
        albedo_views = paint_out["albedo"]
        mr_views     = paint_out["mr"]
        log.metric("albedo_views", len(albedo_views), unit="images")
        log.metric("mr_views",     len(mr_views),     unit="images")

    _save_images(albedo_views, "albedo")
    _save_images(mr_views,     "mr")

    with log.step(f"upscale_views → {cfg.render_size} px"):
        albedo_up = upscale_views(albedo_views, target_size=cfg.render_size)
        mr_up     = upscale_views(mr_views,     target_size=cfg.render_size)
        log.metric("upscaled_px", albedo_up[0].size[0] if albedo_up else 0, unit="px")

    _save_images(albedo_up, "albedo_upscaled")
    _save_images(mr_up,     "mr_upscaled")

    mvd.unload()
    torch.cuda.empty_cache()

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    log.metric("peak_vram_paint_gb", round(peak_vram, 2), unit="GB")

    return albedo_up, mr_up


# ---------------------------------------------------------------------------
# Stage: Bake
# ---------------------------------------------------------------------------

def run_bake(paint_pipeline, albedo_up, mr_up, cfg):
    """Step M: UV texture bake."""
    import torch
    from src.bake_texture import bake_multiview_textures

    print("\n[bake] Cosine-weighted UV back-projection ...")

    with log.step("bake_multiview_textures (albedo + MR)"):
        texture_albedo, mask_albedo, texture_mr, mask_mr = bake_multiview_textures(
            paint_pipeline, albedo_up, mr_up, cfg, save_dir=OUTPUT_DIR
        )

    # Log coverage stats
    try:
        arr = mask_albedo.squeeze(-1).detach().float().cpu().numpy()
        covered = int((arr > 1e-8).sum())
        total   = int(arr.size)
        log.metric("bake_covered_px",   covered, unit="px")
        log.metric("bake_total_px",     total,   unit="px")
        log.metric("bake_coverage_pct", round(100.0 * covered / total, 1), unit="%")
    except Exception:
        pass

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    log.metric("peak_vram_bake_gb", round(peak_vram, 2), unit="GB")

    return texture_albedo, mask_albedo, texture_mr, mask_mr


# ---------------------------------------------------------------------------
# Stage: Inpaint
# ---------------------------------------------------------------------------

def run_inpaint(paint_pipeline, texture_albedo, mask_albedo, texture_mr, mask_mr, cfg):
    """Step N: vertex-aware + cv2 inpainting."""
    import torch
    from src.inpaint_texture import inpaint_textures

    print(
        f"\n[inpaint] Filling uncovered texels  "
        f"vertex={cfg.vertex_inpaint}  method={cfg.inpaint_method} ..."
    )

    with log.step(
        f"inpaint_textures (vertex={cfg.vertex_inpaint}  method={cfg.inpaint_method})"
    ):
        refined_albedo, refined_mr = inpaint_textures(
            paint_pipeline,
            texture_albedo, mask_albedo,
            texture_mr, mask_mr,
            vertex_inpaint=cfg.vertex_inpaint,
            method=cfg.inpaint_method,
            save_dir=OUTPUT_DIR,
        )

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    log.metric("peak_vram_inpaint_gb", round(peak_vram, 2), unit="GB")

    return refined_albedo, refined_mr


# ---------------------------------------------------------------------------
# Stage: Export
# ---------------------------------------------------------------------------

def run_export(paint_pipeline, refined_albedo, refined_mr, uv_mesh):
    """Step O: PBR GLB export."""
    from src.export_glb import export_textured_mesh

    output_glb = OUTPUT_DIR / "output.glb"
    print(f"\n[export] Exporting textured GLB to {output_glb} ...")

    with log.step("export_textured_mesh"):
        glb_path = export_textured_mesh(
            paint_pipeline,
            refined_albedo, refined_mr,
            uv_mesh,
            output_glb,
            cleanup_obj=True,
        )
        log.metric("output_glb", str(glb_path))
        size_mb = glb_path.stat().st_size / 1024 ** 2
        log.metric("output_glb_size_mb", round(size_mb, 1), unit="MB")

    print(f"      saved → {glb_path.relative_to(_PROJECT_ROOT)}")
    return glb_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5 test: bake_texture + inpaint_texture + export_glb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Full pipeline from a single input image
  python scripts/test_phase5.py --image inputs/object.png

  # From a four-view folder
  python scripts/test_phase5.py --multiview inputs/object/

  # Use existing postprocessed GLB + Phase 4 upscaled views
  python scripts/test_phase5.py \\
      --glb  outputs/test/phase4/.../mesh_postprocessed.glb \\
      --albedo outputs/test/phase4/.../albedo_upscaled_*.png \\
      --mr     outputs/test/phase4/.../mr_upscaled_*.png

  # Skip bake — load pre-baked tensors saved with torch.save()
  python scripts/test_phase5.py \\
      --glb outputs/test/phase4/.../mesh_postprocessed.glb \\
      --albedo ... --mr ... \\
      --skip-bake \\
      --baked-albedo outputs/.../baked_albedo.pt \\
      --baked-mr    outputs/.../baked_mr.pt
""",
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--image", type=Path, default=None,
                           help="Single front-view image — full pipeline from scratch.")
    src_group.add_argument("--multiview", type=Path, default=None, metavar="DIR",
                           help="Folder with 4 orientation-suffixed images.")
    src_group.add_argument("--glb", type=Path, default=None,
                           help="Pre-generated postprocessed GLB — skip Phase 3.")

    parser.add_argument("--albedo", nargs="+", type=str, default=None, metavar="IMG",
                        help="Upscaled albedo view PNGs (required with --glb).")
    parser.add_argument("--mr", nargs="+", type=str, default=None, metavar="IMG",
                        help="Upscaled MR view PNGs (required with --glb when not --skip-bake).")
    parser.add_argument("--ref", type=Path, default=None,
                        help="Front-view reference image (required with --glb for paint).")

    parser.add_argument("--shape-steps",  type=int, default=50,  help="Shape steps (default 50).")
    parser.add_argument("--paint-steps",  type=int, default=10,  help="Paint steps (default 10).")
    parser.add_argument("--texture-size", type=int, default=4096, help="UV atlas size (default 4096).")
    parser.add_argument("--seed",         type=int, default=None, help="Random seed.")
    parser.add_argument("--inpaint-method", choices=["NS", "TELEA"], default="NS",
                        help="OpenCV inpaint method (default NS).")
    parser.add_argument("--no-vertex-inpaint", action="store_true",
                        help="Disable vertex-aware inpaint pass (cv2 only).")

    args = parser.parse_args()

    seed = resolve_seed(args.seed)
    log.metric("seed", seed, note="pass --seed to reproduce")
    print(f"Output directory: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")

    from src.config import PipelineConfig
    cfg = PipelineConfig.for_runpod(texture_size=args.texture_size)
    cfg.shape_steps   = args.shape_steps
    cfg.paint_steps   = args.paint_steps
    cfg.seed          = seed
    cfg.inpaint_method = args.inpaint_method
    cfg.vertex_inpaint = not args.no_vertex_inpaint

    failed = False
    try:
        uv_mesh, paint_pipeline, albedo_up, mr_up = build_pipeline_and_mesh(args, cfg)

        texture_albedo, mask_albedo, texture_mr, mask_mr = run_bake(
            paint_pipeline, albedo_up, mr_up, cfg
        )

        refined_albedo, refined_mr = run_inpaint(
            paint_pipeline,
            texture_albedo, mask_albedo,
            texture_mr, mask_mr,
            cfg,
        )

        run_export(paint_pipeline, refined_albedo, refined_mr, uv_mesh)

    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        traceback.print_exc()
        failed = True

    log.save()
    if failed:
        sys.exit(1)
    else:
        print("\nPhase 5 pipeline completed successfully.")


if __name__ == "__main__":
    main()
