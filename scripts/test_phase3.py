"""
Phase 3 test — mesh_generate.py + mesh_postprocess.py

Runs the mesh generation and postprocessing pipeline in sequence.
Requires a GPU with the shape model downloaded.

Outputs written to  outputs/test/phase3/<timestamp>/:
  mesh_raw.glb           — direct from shape pipeline
  mesh_postprocessed.glb — after cleanup + decimation

Usage:
  # Path A — single image (v2.1 DiT)
  python scripts/test_phase3.py --image inputs/object.png

  # Path B — four-view folder (2mv DiT)
  #   folder must contain files ending -front/left/right/back.<ext>
  python scripts/test_phase3.py --multiview inputs/object/
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

from scripts.test_utils import RunLogger, resolve_seed

OUTPUT_DIR = (
    _PROJECT_ROOT / "outputs" / "test" / "phase3"
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log = RunLogger(OUTPUT_DIR, phase="Phase 3")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args, cfg) -> None:
    import torch
    from src.preprocess import (
        load_image_rgba, compose_over_white,
        scan_multiview_folder, collect_views, preprocess_all_views,
    )
    from src.mesh_generate import load_shape_pipeline_auto, generate_mesh
    from src.mesh_postprocess import postprocess_mesh, save_mesh

    # ---- Build shape_views dict --------------------------------------------
    if args.multiview:
        with log.step("scan_multiview_folder"):
            view_paths = scan_multiview_folder(args.multiview)
            for orient, p in view_paths.items():
                log.metric(f"input_{orient}", p.name)

        with log.step("load + preprocess all views"):
            raw_views = collect_views(**{k: str(v) for k, v in view_paths.items()})
            processed = preprocess_all_views(raw_views, remove_bg=False)
            shape_views = {k: v["shape"] for k, v in processed.items()}
            for orient, img in shape_views.items():
                log.metric(f"view_{orient}_size", f"{img.size[0]}×{img.size[1]}", unit="px")

        cfg.use_multiview_shape = True
        mode_label = f"multiview ({len(shape_views)} views)"

    else:
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

    save_mesh(raw_mesh, OUTPUT_DIR / "mesh_raw.glb")
    print(f"      saved → outputs/test/phase3/mesh_raw.glb")

    # ---- Postprocess -------------------------------------------------------
    with log.step(f"postprocess_mesh (target={cfg.target_faces})"):
        post_mesh = postprocess_mesh(
            raw_mesh, target_faces=cfg.target_faces, normalize=cfg.normalize_mesh
        )
        log.metric("post_vertices", len(post_mesh.vertices), unit="verts")
        log.metric("post_faces",    len(post_mesh.faces),    unit="faces")

    save_mesh(post_mesh, OUTPUT_DIR / "mesh_postprocessed.glb")
    print(f"      saved → outputs/test/phase3/mesh_postprocessed.glb")

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    log.metric("peak_vram_gb", round(peak_vram, 2), unit="GB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 test: mesh_generate + mesh_postprocess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/test_phase3.py --image inputs/object.png
  python scripts/test_phase3.py --multiview inputs/object/
        """,
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--image", type=Path, default=None,
        help="Single front-view image → Path A (v2.1 single-image DiT).",
    )
    src_group.add_argument(
        "--multiview", type=Path, default=None,
        metavar="DIR",
        help=(
            "Folder with 4 orientation-suffixed images → Path B (2mv DiT). "
            "Expected filenames: *-front.<ext>  *-left.<ext>  *-right.<ext>  *-back.<ext>"
        ),
    )
    parser.add_argument("--shape-steps", type=int, default=50,
                        help="Shape diffusion steps (default 50).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (logged for reproducibility).")
    args = parser.parse_args()

    seed = resolve_seed(args.seed)
    log.metric("seed", seed, note="pass --seed to reproduce")
    print(f"Output directory: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")
    print(f"Mode: {'multiview' if args.multiview else 'single-image'}")

    from src.config import PipelineConfig
    cfg = PipelineConfig.for_runpod()
    cfg.shape_steps = args.shape_steps
    cfg.seed = seed

    failed = False
    try:
        run_pipeline(args, cfg)
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        traceback.print_exc()
        failed = True

    log.save()
    if failed:
        sys.exit(1)
    else:
        print("\nPhase 3 pipeline completed.")


if __name__ == "__main__":
    main()
