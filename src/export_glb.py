"""
PBR GLB export — plan Step O.

Applies inpainted albedo and metallic-roughness textures to the renderer's
mesh and exports a GLB file with PBR materials.

Export flow:
  1. Load the UV-unwrapped mesh into the renderer (same instance used for bake,
     ensuring UV coordinates are consistent).
  2. Set albedo texture via render.set_texture(refined_albedo, force_set=True).
  3. Set MR texture via render.set_texture_mr(refined_mr).
  4. Save the mesh + textures as an OBJ with companion JPEG texture files via
     render.save_mesh().  The JPEG files are named:
       <stem>.jpg           — albedo map
       <stem>_metallic.jpg  — metallic channel
       <stem>_roughness.jpg — roughness channel
  5. Convert OBJ + textures to a GLB with full PBR material definitions via
     create_glb_with_pbr_materials() from hy3dpaint's convert_utils module.
  6. Optionally clean up the intermediate OBJ and texture files.

Critical constraints (plan Section 11):
  - The SAME mesh instance loaded into the renderer at step H must be reloaded
    here.  Reusing a different copy with the same geometry still causes a UV
    mismatch in the renderer's internal state.
  - Texture files must exist next to the OBJ when create_glb_with_pbr_materials
    is called — the function looks them up by filename.
  - downsample=False must be passed to save_mesh to preserve full texture
    resolution.

Import note: convert_utils and DifferentiableRenderer are only importable on
RunPod (they require the compiled CUDA extensions).  All heavy imports are
deferred to the function body.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import trimesh

if TYPE_CHECKING:
    from src.render_multiview import PaintPipeline  # type: ignore[import]

logger = logging.getLogger(__name__)


def export_textured_mesh(
    pipeline: "PaintPipeline",
    refined_albedo,
    refined_mr,
    mesh: trimesh.Trimesh,
    output_path: "Path | str",
    *,
    cleanup_obj: bool = True,
) -> Path:
    """
    Apply textures to the mesh and export as a PBR GLB.

    Args:
        pipeline:       Initialized PaintPipeline.  The mesh is (re)loaded into
                        the renderer here to ensure UV state is consistent with
                        the bake pass.
        refined_albedo: Inpainted albedo texture from inpaint_textures().
        refined_mr:     Inpainted metallic-roughness texture.
        mesh:           The UV-unwrapped trimesh.Trimesh used throughout the
                        pipeline.  Must be the same geometry instance as was
                        used during the bake stage.
        output_path:    Desired output file path.  The .glb extension is always
                        used for the final output regardless of the suffix
                        provided.  The OBJ and JPEG intermediates are placed in
                        the same directory with the same stem.
        cleanup_obj:    If True (default), remove the intermediate OBJ and
                        JPEG texture files after GLB export succeeds.

    Returns:
        Path to the exported GLB file.

    Raises:
        ImportError:   If convert_utils is not importable (CUDA extensions not
                       compiled — expected on macOS dev machines).
        RuntimeError:  If GLB export fails for any other reason.
    """
    from convert_utils import create_glb_with_pbr_materials  # type: ignore[import]

    output_path = Path(output_path).with_suffix(".glb")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stem = output_path.stem
    output_dir = output_path.parent
    output_obj = output_dir / f"{stem}.obj"
    output_glb = output_path

    texture_paths = {
        "albedo":     str(output_dir / f"{stem}.jpg"),
        "metallic":   str(output_dir / f"{stem}_metallic.jpg"),
        "roughness":  str(output_dir / f"{stem}_roughness.jpg"),
    }

    logger.info("Exporting textured mesh to %s ...", output_glb)

    # (Re)load the mesh into the renderer so its internal UV state matches the
    # bake pass.  This is a lightweight operation — it just updates the GPU
    # buffers, no geometry recomputation.
    pipeline.load_mesh(mesh)

    pipeline.render.set_texture(refined_albedo, force_set=True)
    pipeline.render.set_texture_mr(refined_mr)

    logger.info("Saving OBJ + textures to %s ...", output_obj)
    pipeline.render.save_mesh(str(output_obj), downsample=False)

    logger.info("Converting OBJ to GLB with PBR materials ...")
    create_glb_with_pbr_materials(str(output_obj), texture_paths, str(output_glb))

    if not output_glb.exists():
        raise RuntimeError(
            f"GLB export appeared to succeed but output file not found: {output_glb}"
        )

    glb_size_mb = output_glb.stat().st_size / 1024 ** 2
    logger.info("GLB exported: %s  (%.1f MB)", output_glb, glb_size_mb)

    if cleanup_obj:
        _cleanup_obj_files(output_obj, texture_paths)

    return output_glb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup_obj_files(
    output_obj: Path,
    texture_paths: dict,
) -> None:
    """Remove intermediate OBJ, MTL, and JPEG texture files."""
    candidates = [
        output_obj,
        output_obj.with_suffix(".mtl"),
        *[Path(p) for p in texture_paths.values()],
    ]
    for p in candidates:
        try:
            if p.exists():
                p.unlink()
        except OSError as exc:
            logger.warning("Could not remove intermediate file %s: %s", p, exc)
    logger.debug("Removed intermediate OBJ/texture files.")
