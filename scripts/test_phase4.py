"""
Phase 4 test — render_multiview.py + paint_multiview.py

Runs UV unwrap, conditioning map rendering, and multiview paint diffusion
in sequence. Requires a GPU with the custom_rasterizer extension compiled.
The paint diffusion stage is skipped if the model weights are not present.

Outputs written to  outputs/test/phase4/<timestamp>/:
  mesh_uv.glb              — UV-unwrapped mesh
  normal_00.png …          — 6 normal conditioning maps
  position_00.png …        — 6 position conditioning maps
  reference_delighted.png  — delight reference (if delight model present)
  albedo_00.png …          — 6 diffusion albedo views (if paint model present)
  mr_00.png …              — 6 metallic-roughness views
  albedo_upscaled_00.png … — Lanczos upscaled albedo
  mr_upscaled_00.png …     — Lanczos upscaled MR

Usage:
  python scripts/test_phase4.py --image path/to/input.png [options]
  python scripts/test_phase4.py --glb path/to/mesh_postprocessed.glb [options]

  --image          Input image — generates mesh via Phase 3 pipeline first.
  --glb            Pre-generated postprocessed GLB (skips mesh generation).
  --shape-steps    Shape diffusion steps when using --image (default 50).
  --paint-steps    Paint diffusion steps (default 10).
  --skip-paint     Skip MultiviewDiffusionNet even if models are present.
  --seed           Random seed (logged for reproducibility).
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

import trimesh
from scripts.test_utils import RunLogger, resolve_seed

OUTPUT_DIR = (
    _PROJECT_ROOT / "outputs" / "test" / "phase4"
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log = RunLogger(OUTPUT_DIR, phase="Phase 4")


def save_images(images, prefix: str) -> None:
    for i, img in enumerate(images):
        img.save(str(OUTPUT_DIR / f"{prefix}_{i:02d}.png"))
    print(f"      saved {len(images)} → {prefix}_*.png")


# ---------------------------------------------------------------------------
# Load or generate mesh
# ---------------------------------------------------------------------------

def load_postprocessed_mesh(args, cfg) -> trimesh.Trimesh:
    if args.glb:
        print(f"\n[mesh] Loading from {args.glb}")
        loaded = trimesh.load(str(args.glb))
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        log.metric("mesh_source", str(args.glb))
        return loaded

    from src.preprocess import (
        load_image_rgba, compose_over_white,
        scan_multiview_folder, collect_views, preprocess_all_views,
    )
    from src.mesh_generate import load_shape_pipeline_auto, generate_mesh
    from src.mesh_postprocess import postprocess_mesh, save_mesh

    # ---- Build shape_views dict --------------------------------------------
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
        mode_label = f"multiview ({len(shape_views)} views)"

    else:
        print(f"\n[mesh] Generating from {args.image}")

        with log.step("load_image + white composite"):
            raw_img = load_image_rgba(args.image)
            shape_views = {"front": compose_over_white(raw_img)}
            log.metric("input_image", f"{args.image.name} {raw_img.size[0]}×{raw_img.size[1]}", unit="px")

        cfg.use_multiview_shape = False
        mode_label = "single-image"

    log.metric("generation_mode", mode_label)

    # ---- Load pipeline + generate ------------------------------------------
    with log.step("load_shape_pipeline_auto"):
        pipeline = load_shape_pipeline_auto(cfg)

    with log.step(f"generate_mesh ({mode_label}, steps={cfg.shape_steps})"):
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
    return post_mesh


# ---------------------------------------------------------------------------
# Render stage
# ---------------------------------------------------------------------------

def run_render(args, cfg) -> tuple:
    import torch
    from src.render_multiview import uv_unwrap_mesh, PaintPipeline, render_conditioning_maps

    print("\n[render] UV unwrap + conditioning maps")

    post_mesh = load_postprocessed_mesh(args, cfg)

    with log.step("uv_unwrap_mesh"):
        uv_mesh = uv_unwrap_mesh(post_mesh)
        log.metric("uv_vertices", len(uv_mesh.vertices), unit="verts")
        log.metric("uv_faces",    len(uv_mesh.faces),    unit="faces")

    uv_mesh.export(str(OUTPUT_DIR / "mesh_uv.glb"))
    print("      saved → mesh_uv.glb")

    with log.step("PaintPipeline init"):
        paint_pipeline = PaintPipeline(cfg)

    with log.step("load_mesh into renderer"):
        paint_pipeline.load_mesh(uv_mesh)

    with log.step(f"render_conditioning_maps ({len(cfg.camera.azimuths)} views)"):
        normal_maps, position_maps = render_conditioning_maps(paint_pipeline, cfg)
        log.metric("normal_maps",   len(normal_maps),   unit="images")
        log.metric("position_maps", len(position_maps), unit="images")

    save_images(normal_maps,   "normal")
    save_images(position_maps, "position")

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    log.metric("peak_vram_render_gb", round(peak_vram, 2), unit="GB")

    return uv_mesh, paint_pipeline, normal_maps, position_maps


# ---------------------------------------------------------------------------
# Paint stage
# ---------------------------------------------------------------------------

def run_paint(args, cfg, uv_mesh, paint_pipeline, normal_maps, position_maps) -> None:
    import torch
    from src.paint_multiview import MultiviewDiffusionNet, delight_reference, upscale_views
    from src.preprocess import load_image_rgba, scan_multiview_folder

    paint_model_dir = cfg.models_dir / "hunyuan3d-paintpbr-v2-1"
    if not paint_model_dir.exists():
        print(
            f"\n[paint] SKIPPED — model not found at {paint_model_dir}.\n"
            "  Run:  python scripts/download_models.py"
        )
        return

    # Resolve the front-view reference image path.
    # For multiview, use the -front image from the scanned folder.
    if args.image:
        front_path = args.image
    elif args.multiview:
        view_paths = scan_multiview_folder(args.multiview)
        front_path = view_paths["front"]
    else:
        print("\n[paint] SKIPPED — no reference image available.")
        return

    print("\n[paint] Delight + diffusion + upscale")

    with log.step("delight_reference"):
        front_rgba = load_image_rgba(front_path)
        reference = delight_reference(front_rgba, cfg)
        reference.save(str(OUTPUT_DIR / "reference_delighted.png"))

    with log.step("MultiviewDiffusionNet init"):
        mvd = MultiviewDiffusionNet(cfg)

    with log.step(f"paint diffusion (steps={cfg.paint_steps})"):
        paint_out = mvd(reference, normal_maps, position_maps, cfg)
        albedo_views = paint_out["albedo"]
        mr_views     = paint_out["mr"]
        log.metric("albedo_views", len(albedo_views), unit="images")
        log.metric("mr_views",     len(mr_views),     unit="images")

    save_images(albedo_views, "albedo")
    save_images(mr_views,     "mr")

    with log.step(f"upscale_views → {cfg.render_size}px"):
        albedo_up = upscale_views(albedo_views, target_size=cfg.render_size)
        mr_up     = upscale_views(mr_views,     target_size=cfg.render_size)
        log.metric("upscaled_px", albedo_up[0].size[0] if albedo_up else 0, unit="px")

    save_images(albedo_up, "albedo_upscaled")
    save_images(mr_up,     "mr_upscaled")

    mvd.unload()
    torch.cuda.empty_cache()

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    log.metric("peak_vram_paint_gb", round(peak_vram, 2), unit="GB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 test: render_multiview + paint_multiview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # single image → mesh generation + render + paint
  python scripts/test_phase4.py --image inputs/object.png

  # four-view folder → multiview mesh generation + render + paint
  python scripts/test_phase4.py --multiview inputs/object/

  # skip mesh generation, use existing GLB
  python scripts/test_phase4.py --glb outputs/test/phase3/.../mesh_postprocessed.glb
        """,
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--image", type=Path, default=None,
        help="Single front-view image → Path A (v2.1 DiT) mesh generation.",
    )
    src_group.add_argument(
        "--multiview", type=Path, default=None, metavar="DIR",
        help=(
            "Folder with 4 orientation-suffixed images → Path B (2mv DiT). "
            "Expected: *-front.<ext>  *-left.<ext>  *-right.<ext>  *-back.<ext>"
        ),
    )
    src_group.add_argument(
        "--glb", type=Path, default=None,
        help="Pre-generated postprocessed GLB — skips mesh generation entirely.",
    )
    parser.add_argument("--shape-steps", type=int, default=50,
                        help="Shape diffusion steps when generating mesh (default 50).")
    parser.add_argument("--paint-steps", type=int, default=10,
                        help="Paint diffusion steps (default 10).")
    parser.add_argument("--skip-paint", action="store_true",
                        help="Skip MultiviewDiffusionNet even if models are present.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (logged for reproducibility).")
    args = parser.parse_args()

    # Paint needs a front reference image; auto-skip only when using --glb without --image.
    if not args.skip_paint and args.glb and args.image is None and args.multiview is None:
        print("NOTE: --glb used without --image or --multiview — paint stage will be skipped.")
        args.skip_paint = True

    seed = resolve_seed(args.seed)
    log.metric("seed", seed, note="pass --seed to reproduce")
    print(f"Output directory: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")

    from src.config import PipelineConfig
    cfg = PipelineConfig.for_runpod()
    cfg.shape_steps = args.shape_steps
    cfg.paint_steps = args.paint_steps
    cfg.seed        = seed

    failed = False
    try:
        uv_mesh, paint_pipeline, normal_maps, position_maps = run_render(args, cfg)

        if not args.skip_paint:
            run_paint(args, cfg, uv_mesh, paint_pipeline, normal_maps, position_maps)
        else:
            print("\n[paint] SKIPPED — --skip-paint flag set.")

    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        traceback.print_exc()
        failed = True

    log.save()
    if failed:
        sys.exit(1)
    else:
        print("\nPhase 4 pipeline completed.")


if __name__ == "__main__":
    main()
