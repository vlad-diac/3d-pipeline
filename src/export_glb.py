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
    output_format: str = "glb",
) -> Path:
    """
    Apply textures to the mesh and export as GLB or OBJ.

    Args:
        pipeline:       Initialized PaintPipeline.  The mesh is (re)loaded into
                        the renderer here to ensure UV state is consistent with
                        the bake pass.
        refined_albedo: Inpainted albedo texture from inpaint_textures().
        refined_mr:     Inpainted metallic-roughness texture.
        mesh:           The UV-unwrapped trimesh.Trimesh used throughout the
                        pipeline.  Must be the same geometry instance as was
                        used during the bake stage.
        output_path:    Desired output file path.  The extension is overridden
                        by ``output_format``.  OBJ and JPEG intermediates are
                        placed in the same directory with the same stem.
        cleanup_obj:    If True (default), remove the intermediate OBJ and
                        JPEG texture files after GLB export succeeds.
                        Ignored when output_format is "obj".
        output_format:  "glb" (default) — convert to GLB with PBR materials.
                        "obj" — keep the OBJ + JPEG texture sidecars as the
                        final output (no GLB conversion step).

    Returns:
        Path to the exported file (.glb or .obj).

    Raises:
        ImportError:   If convert_utils is not importable and format is "glb"
                       (CUDA extensions not compiled — expected on macOS).
        RuntimeError:  If export fails for any other reason.
        ValueError:    If output_format is not "glb" or "obj".
    """
    if output_format not in ("glb", "obj"):
        raise ValueError(f"output_format must be 'glb' or 'obj', got {output_format!r}")

    output_path = Path(output_path).with_suffix(f".{output_format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stem = output_path.stem
    output_dir = output_path.parent
    output_obj = output_dir / f"{stem}.obj"

    texture_paths = {
        "albedo":     str(output_dir / f"{stem}.jpg"),
        "metallic":   str(output_dir / f"{stem}_metallic.jpg"),
        "roughness":  str(output_dir / f"{stem}_roughness.jpg"),
    }

    logger.info("Exporting textured mesh → %s (format=%s) ...", output_path, output_format)

    # (Re)load the mesh into the renderer so its internal UV state matches the
    # bake pass.  This is a lightweight operation — it just updates the GPU
    # buffers, no geometry recomputation.
    pipeline.load_mesh(mesh)

    pipeline.render.set_texture(refined_albedo, force_set=True)
    pipeline.render.set_texture_mr(refined_mr)

    # Save OBJ + textures ourselves — MeshRender.save_mesh() calls save_mesh()
    # from mesh_utils.py which imports bpy at the module level and therefore
    # cannot be imported outside a Blender environment.  We replicate the
    # relevant logic here using only numpy / cv2.
    logger.info("Saving OBJ + textures to %s ...", output_obj)
    _save_renderer_mesh_to_obj(pipeline.render, output_obj, stem)

    if output_format == "obj":
        logger.info("OBJ export complete: %s", output_obj)
        return output_obj

    # GLB path: convert OBJ + textures to a self-contained binary glTF.
    from convert_utils import create_glb_with_pbr_materials  # type: ignore[import]

    output_glb = output_path
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

def _save_renderer_mesh_to_obj(render, output_obj: Path, stem: str) -> None:
    """
    Write OBJ + MTL + JPEG texture files from a MeshRender instance.

    Replicates mesh_utils.save_obj_mesh() without the top-level ``import bpy``
    that makes that module unimportable outside a Blender environment.

    The files written alongside the OBJ are:
      <stem>.mtl            — material definition
      <stem>.jpg            — albedo (diffuse) map
      <stem>_metallic.jpg   — metallic channel (greyscale)
      <stem>_roughness.jpg  — roughness channel (greyscale)
    """
    import cv2
    import numpy as np
    from io import StringIO

    output_dir = output_obj.parent
    base_path = str(output_dir / stem)

    # ── 1. Mesh geometry ──────────────────────────────────────────────────
    vtx_pos, pos_idx, vtx_uv, uv_idx = render.get_mesh(normalize=False)
    vtx_pos = np.array(vtx_pos, dtype=np.float32)
    vtx_uv  = np.array(vtx_uv,  dtype=np.float32)
    pos_idx = np.array(pos_idx, dtype=np.int32)
    uv_idx  = np.array(uv_idx,  dtype=np.int32)

    # ── 2. OBJ content ────────────────────────────────────────────────────
    buf = StringIO()
    buf.write(f"mtllib {stem}.mtl\no {stem}\n")
    np.savetxt(buf, vtx_pos, fmt="v %.6f %.6f %.6f")
    np.savetxt(buf, vtx_uv,  fmt="vt %.6f %.6f")
    buf.write("s 0\nusemtl Material\n")

    pos_idx1 = pos_idx + 1
    uv_idx1  = uv_idx  + 1
    fmt_face = np.frompyfunc(lambda p, u: f"{int(p)}/{int(u)}", 2, 1)
    face_strs = [f"f {' '.join(row)}" for row in fmt_face(pos_idx1, uv_idx1)]
    buf.write("\n".join(face_strs) + "\n")
    output_obj.write_text(buf.getvalue())

    # ── 3. Textures ───────────────────────────────────────────────────────
    def _save_tex(arr: np.ndarray, suffix: str = "", greyscale: bool = False) -> str:
        path = f"{base_path}{suffix}.jpg"
        img_u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        if greyscale:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(path, img_u8)
        else:
            cv2.imwrite(path, img_u8[..., ::-1])  # RGB → BGR for cv2
        return path

    albedo = render.get_texture()
    _save_tex(albedo)

    metallic, roughness = render.get_texture_mr()
    has_mr = metallic is not None and roughness is not None
    if has_mr:
        _save_tex(metallic,  "_metallic",  greyscale=True)
        _save_tex(roughness, "_roughness", greyscale=True)

    # ── 4. MTL file ───────────────────────────────────────────────────────
    mtl_path = f"{base_path}.mtl"
    with open(mtl_path, "w") as f:
        f.write("newmtl Material\n")
        f.write(f"Kd 0.800 0.800 0.800\n")
        f.write(f"Ke 0.000 0.000 0.000\n")
        f.write(f"Ni 1.500\n")
        f.write(f"d 1.0\n")
        f.write(f"illum {'2' if has_mr else '3'}\n")
        f.write(f"map_Kd {stem}.jpg\n")
        if has_mr:
            f.write(f"map_Pm {stem}_metallic.jpg\n")
            f.write(f"map_Pr {stem}_roughness.jpg\n")


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
