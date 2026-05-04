#!/usr/bin/env python3
"""
download_models.py — Download Hunyuan3D model weights from HuggingFace Hub.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models-dir /path/to/models
    python scripts/download_models.py --skip-existing
    python scripts/download_models.py --models multiview_shape dinov2
    python scripts/download_models.py --dry-run

Models downloaded (by key):
    shape_v21         tencent/Hunyuan3D-2.1   subfolder: hunyuan3d-dit-v2-1       (~7.4 GB)
    vae_v21           tencent/Hunyuan3D-2.1   subfolder: hunyuan3d-vae-v2-1       (~656 MB)
    paint_pbr_v21     tencent/Hunyuan3D-2.1   subfolder: hunyuan3d-paintpbr-v2-1  (~4 GB)
    dinov2            facebook/dinov2-giant                                        (~4.4 GB)
    multiview_shape   tencent/Hunyuan3D-2mv   subfolder: hunyuan3d-dit-v2-mv      (~7 GB)   [optional]
    realesrgan        direct URL (GitHub)                                          (~64 MB)

The delight model (Light_Shadow_Remover) is part of the Hunyuan3D-2 repo and does
not have a standalone HuggingFace entry yet — it must be obtained from the cloned
third_party/Hunyuan3D-2 release assets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ─── Model registry ──────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "shape_v21": {
        "type": "snapshot",
        "repo_id": "tencent/Hunyuan3D-2.1",
        "allow_patterns": ["hunyuan3d-dit-v2-1/**"],
        "local_dir_suffix": "hunyuan3d-dit-v2-1",
        "description": "Shape DiT v2.1 (~7.4 GB)",
        "required": True,
    },
    "vae_v21": {
        "type": "snapshot",
        "repo_id": "tencent/Hunyuan3D-2.1",
        "allow_patterns": ["hunyuan3d-vae-v2-1/**"],
        "local_dir_suffix": "hunyuan3d-vae-v2-1",
        "description": "VAE v2.1 (~656 MB)",
        "required": True,
    },
    "paint_pbr_v21": {
        "type": "snapshot",
        "repo_id": "tencent/Hunyuan3D-2.1",
        "allow_patterns": ["hunyuan3d-paintpbr-v2-1/**"],
        "local_dir_suffix": "hunyuan3d-paintpbr-v2-1",
        "description": "Paint-PBR v2.1 (~4 GB)",
        "required": True,
    },
    "dinov2": {
        "type": "snapshot",
        "repo_id": "facebook/dinov2-giant",
        "allow_patterns": None,  # full repo
        "local_dir_suffix": "dinov2-giant",
        "description": "DINOv2-Giant (~4.4 GB)",
        "required": True,
    },
    "multiview_shape": {
        "type": "snapshot",
        "repo_id": "tencent/Hunyuan3D-2mv",
        "allow_patterns": ["hunyuan3d-dit-v2-mv/**"],
        "local_dir_suffix": "hunyuan3d-dit-v2-mv",
        "description": "4-View Shape DiT v2.0 (~7 GB) [optional, Path B only]",
        "required": False,
    },
    "realesrgan": {
        "type": "url",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "local_filename": "RealESRGAN_x4plus.pth",
        "description": "RealESRGAN x4plus upscaler (~64 MB)",
        "required": True,
    },
}

# Required model keys downloaded by default (multiview_shape is optional)
DEFAULT_MODELS = [k for k, v in MODELS.items() if v["required"]]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _check_skip(dest: Path, skip_existing: bool) -> bool:
    """Return True if the destination already exists and we should skip."""
    if not skip_existing:
        return False
    if dest.is_dir() and any(dest.iterdir()):
        return True
    if dest.is_file():
        return True
    return False


def download_snapshot(
    model_key: str,
    spec: dict,
    models_dir: Path,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    dest = models_dir / spec["local_dir_suffix"]
    print(f"\n[{model_key}] {spec['description']}")
    print(f"  → {dest}")

    if _check_skip(dest, skip_existing):
        print("  SKIP (already exists)")
        return

    if dry_run:
        print("  DRY-RUN — would call snapshot_download")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {
        "repo_id": spec["repo_id"],
        "local_dir": str(dest),
        "local_dir_use_symlinks": False,
    }
    if spec.get("allow_patterns"):
        kwargs["allow_patterns"] = spec["allow_patterns"]

    snapshot_download(**kwargs)
    print(f"  OK")


def download_url(
    model_key: str,
    spec: dict,
    models_dir: Path,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    dest = models_dir / spec["local_filename"]
    print(f"\n[{model_key}] {spec['description']}")
    print(f"  → {dest}")

    if _check_skip(dest, skip_existing):
        print("  SKIP (already exists)")
        return

    if dry_run:
        print("  DRY-RUN — would download from URL")
        return

    import urllib.request

    models_dir.mkdir(parents=True, exist_ok=True)

    def _progress(count: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            print(f"\r  Downloading... {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(spec["url"], dest, reporthook=_progress)
        print(f"\r  OK{' ' * 20}")
    except Exception as exc:
        print(f"\n  ERROR downloading {spec['url']}: {exc}")
        sys.exit(1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hunyuan3D model weights from HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Directory to save model weights (default: <project_root>/models/)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip models whose destination directory/file already exists (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-download even if destination exists",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODELS,
        metavar="MODEL_KEY",
        help=(
            f"Which models to download. Choices: {', '.join(MODELS.keys())}. "
            f"Default: all required models ({', '.join(DEFAULT_MODELS)})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir: Path = args.models_dir.resolve()

    print(f"Models directory : {models_dir}")
    print(f"Skip existing    : {args.skip_existing}")
    print(f"Dry run          : {args.dry_run}")
    print(f"Models requested : {', '.join(args.models)}")

    models_dir.mkdir(parents=True, exist_ok=True)

    for key in args.models:
        spec = MODELS[key]
        if spec["type"] == "snapshot":
            download_snapshot(key, spec, models_dir, args.skip_existing, args.dry_run)
        elif spec["type"] == "url":
            download_url(key, spec, models_dir, args.skip_existing, args.dry_run)
        else:
            print(f"\n[{key}] Unknown download type: {spec['type']}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
