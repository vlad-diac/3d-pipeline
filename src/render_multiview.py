"""
UV unwrap + paint pipeline initialization + conditioning map rendering.

Covers plan Steps F, H, and I:
  F — UV unwrap the postprocessed mesh using xatlas.
  H — Initialize the paint pipeline: MeshRender + ViewProcessor.
  I — Render normal maps and position maps from each of the 6 camera views.

These outputs feed directly into paint_multiview.py (Steps J–L) and
bake_texture.py (Step M).

Import note: All hy3dpaint imports are deferred to function/class bodies.
src/__init__.py has already inserted third_party paths into sys.path.
This module is importable on macOS (no GPU call at import time).

Critical constraints (from the plan):
  - UV unwrap MUST happen before loading the mesh into MeshRender.  The
    renderer projects colours into UV coordinates, so UVs must exist on
    the mesh before load_mesh() is called.
  - View order must remain consistent through render → paint → bake stages.
    Changing the order produces mirrored or shifted baked textures.
  - ortho_scale must be set on MeshRender after construction via
    set_orth_scale().  The constructor hardcodes 1.2; call set_orth_scale()
    to override (cfg.camera.ortho_scale=1.0 to fit the normalized mesh).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import trimesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step F: UV unwrap
# ---------------------------------------------------------------------------

def uv_unwrap_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Apply xatlas UV parametrization to the mesh.

    Must be called after postprocess_mesh() and before rendering or baking.
    xatlas generates a new vertex/index buffer (vmapping may split vertices
    at UV seams), so the returned mesh typically has more vertices than the
    input.

    Args:
        mesh: Postprocessed trimesh.Trimesh (no UVs required on input).

    Returns:
        New trimesh.Trimesh with .visual.uv set — ready for MeshRender.

    Raises:
        ValueError: If the mesh exceeds the xatlas 500M-face hard limit.
    """
    import xatlas  # type: ignore[import]

    if isinstance(mesh, trimesh.Scene):
        logger.debug("uv_unwrap_mesh: received Scene — concatenating.")
        mesh = mesh.dump(concatenate=True)

    n_verts_in = len(mesh.vertices)
    n_faces_in = len(mesh.faces)

    if n_faces_in > 500_000_000:
        raise ValueError(
            f"Mesh has {n_faces_in:,} faces — exceeds xatlas limit of 500M."
        )

    logger.info(
        "UV unwrapping mesh: %d vertices, %d faces.", n_verts_in, n_faces_in
    )

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    mesh = mesh.copy()
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    logger.info(
        "UV unwrap complete: %d vertices (was %d), UV atlas size: %d coords.",
        len(mesh.vertices), n_verts_in, len(uvs),
    )
    return mesh


# ---------------------------------------------------------------------------
# Step H: Paint pipeline — MeshRender + ViewProcessor
# ---------------------------------------------------------------------------

class PaintPipeline:
    """
    Container for the mesh renderer and view-processing components.

    Initializes:
      - MeshRender: GPU rasterizer for normal/position maps and UV baking.
      - ViewProcessor: wraps MeshRender for multiview render + bake operations.

    The multiview diffusion model (MultiviewDiffusionNet in paint_multiview.py)
    is NOT held here — it is loaded and freed separately to manage VRAM.
    The paint model uses ~21 GB; MeshRender itself is lightweight.

    Notes on MeshRender construction:
      - raster_mode="cr" imports the compiled custom_rasterizer CUDA extension.
        This means PaintPipeline can ONLY be instantiated on RunPod (GPU).
      - ortho_scale defaults to 1.2 inside MeshRender.  We override it to
        cfg.camera.ortho_scale (default 1.0) so the orthographic frustum fits
        the normalized unit mesh produced by postprocess_mesh().

    Args:
        cfg: PipelineConfig with render_size, texture_size, camera, device.
    """

    def __init__(self, cfg) -> None:
        from DifferentiableRenderer.MeshRender import MeshRender  # type: ignore[import]
        from utils.pipeline_utils import ViewProcessor  # type: ignore[import]

        self.cfg = cfg

        logger.info(
            "Initializing MeshRender: render_size=%d  texture_size=%d  ortho_scale=%.2f",
            cfg.render_size, cfg.texture_size, cfg.camera.ortho_scale,
        )

        self.render = MeshRender(
            default_resolution=cfg.render_size,
            texture_size=cfg.texture_size,
            bake_mode="back_sample",
            raster_mode="cr",
            device=cfg.device,
        )

        # Override the hard-coded 1.2 ortho scale set in MeshRender.__init__.
        # ortho_scale=1.0 ensures the frustum covers the normalized unit mesh exactly.
        self.render.set_orth_scale(cfg.camera.ortho_scale)

        # ViewProcessor uses config.bake_exp for the cosine-weighted bake formula:
        #   blend_weight = view_weight × cos(θ)^bake_exp
        # bake_exp=4 concentrates trust on nearly face-on views (plan default).
        vp_cfg = SimpleNamespace(
            device=cfg.device,
            render_size=cfg.render_size,
            texture_size=cfg.texture_size,
            bake_exp=4,
            merge_method="fast",
            raster_mode="cr",
            bake_mode="back_sample",
            resolution=cfg.view_size,
        )

        self.view_processor = ViewProcessor(vp_cfg, self.render)
        logger.info("PaintPipeline initialized.")

    def load_mesh(self, mesh: trimesh.Trimesh) -> None:
        """
        Load the UV-unwrapped mesh into the renderer.

        Must be called with the SAME mesh instance used for UV baking and
        export — the renderer's internal state (UV buffer, vertex positions)
        must remain consistent across render → bake → export stages.

        Calls MeshRender.set_mesh() directly with numpy arrays extracted from
        the trimesh, bypassing MeshRender.load_mesh() which requires the bpy
        mesh_utils helper (not available without Blender).

        After xatlas UV unwrap, faces index both geometry and UV consistently,
        so pos_idx == uv_idx.

        Args:
            mesh: UV-unwrapped trimesh.Trimesh with .visual.uv set.
        """
        import numpy as np

        vtx_pos = np.array(mesh.vertices, dtype=np.float32)
        pos_idx = np.array(mesh.faces,    dtype=np.int32)

        vtx_uv: "np.ndarray | None" = None
        uv_idx: "np.ndarray | None" = None
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            vtx_uv = np.array(mesh.visual.uv, dtype=np.float32)
            uv_idx = pos_idx.copy()

        self.render.set_mesh(vtx_pos, pos_idx, vtx_uv=vtx_uv, uv_idx=uv_idx)
        logger.info(
            "Mesh loaded into renderer: %d vertices, %d faces, UVs=%s.",
            len(mesh.vertices), len(mesh.faces), vtx_uv is not None,
        )


# ---------------------------------------------------------------------------
# Step I: Render conditioning maps
# ---------------------------------------------------------------------------

def render_conditioning_maps(
    pipeline: PaintPipeline,
    cfg,
) -> Tuple[List, List]:
    """
    Render normal maps and position maps for all camera views.

    Normal maps encode surface orientation (Nx, Ny, Nz → RGB).  They tell
    the paint model where edges and creases are, enabling view-consistent
    shading.

    Position maps encode world-space coordinates (normalized → RGB).  They
    ensure the model understands which part of the surface each pixel
    corresponds to, maintaining geometric consistency across views.

    The condition tensor for the paint pipeline is structured as:
        [normal_0, ..., normal_N, position_0, ..., position_N]

    This ordering is critical: the pipeline splits at len//2 to separate
    normals from positions.  The same ordering must be used when calling
    MultiviewDiffusionNet in paint_multiview.py.

    Args:
        pipeline: Initialized PaintPipeline with mesh already loaded.
        cfg:      PipelineConfig — uses cfg.camera.elevations / .azimuths.

    Returns:
        (normal_maps, position_maps) — each a list of PIL.Image objects,
        one per camera view, in the same order as cfg.camera.azimuths.
    """
    n_views = len(cfg.camera.azimuths)
    logger.info(
        "Rendering conditioning maps for %d views (elevations=%s  azimuths=%s).",
        n_views, cfg.camera.elevations, cfg.camera.azimuths,
    )

    normal_maps: List = pipeline.view_processor.render_normal_multiview(
        cfg.camera.elevations,
        cfg.camera.azimuths,
        use_abs_coor=True,
    )

    position_maps: List = pipeline.view_processor.render_position_multiview(
        cfg.camera.elevations,
        cfg.camera.azimuths,
    )

    logger.info(
        "Conditioning maps done: %d normals, %d positions.",
        len(normal_maps), len(position_maps),
    )
    return normal_maps, position_maps
