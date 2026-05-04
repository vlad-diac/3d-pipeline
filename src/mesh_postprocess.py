"""
Mesh postprocessing — plan Step E.

Cleans geometry artifacts produced by marching cubes and normalizes the mesh
for consistent downstream renderer behavior.

Processing chain (in order):
  1. remove_infinite_values()       — catches NaN/Inf from MC artifacts
  2. remove_unreferenced_vertices() — purge orphan verts
  3. remove_duplicate_faces()       — remove exact duplicates
  4. remove_degenerate_faces()      — zero-area triangles
  5. merge_vertices()               — weld coincident verts
  6. keep_largest_component()       — drop floaters / disconnected fragments
  7. decimate()                     — reduce to target_faces (trimesh → pymeshlab fallback)
  8. normalize_mesh()               — center at origin, scale to unit bounding box

Key decisions from the plan:
  - 200k face target (Research 1 + 2: Tutorial's 40k is too aggressive for texture quality).
  - Fallback decimation chain: trimesh built-in → pymeshlab.
  - process=False on trimesh construction (prevents unwanted geometry modification).
  - remove_infinite_values() first (Research 2: catches MC artifacts).
  - normalize_mesh() last so scale is predictable for all downstream stages.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Return only the largest connected component (by face count).

    Removes floating fragments (floaters) that marching cubes sometimes
    produces from noisy signed-distance fields.
    """
    parts = mesh.split(only_watertight=False)
    if not parts:
        logger.warning("keep_largest_component: mesh split returned no parts.")
        return mesh
    largest = max(parts, key=lambda m: len(m.faces))
    n_removed = len(mesh.faces) - len(largest.faces)
    if n_removed > 0:
        logger.debug(
            "Removed %d floater faces (%d components → 1).",
            n_removed, len(parts),
        )
    return largest


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Center the mesh at the origin and scale so the longest bounding-box
    dimension equals 1.0.

    Normalization ensures the renderer's orthographic camera frustum
    (ortho_scale=1.0) covers the mesh exactly, regardless of the raw
    scale that came out of the shape model.
    """
    mesh = mesh.copy()
    bounds = mesh.bounds  # shape (2, 3)
    center = bounds.mean(axis=0)
    scale = float(np.max(bounds[1] - bounds[0]))

    if scale <= 1e-8:
        logger.warning("normalize_mesh: bounding box scale is near-zero (%.2e). Skipping.", scale)
        return mesh

    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale)
    logger.debug("Normalized mesh: center=%s → origin, scale=%.4f → 1.0.", center, scale)
    return mesh


def decimate_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int,
) -> trimesh.Trimesh:
    """
    Reduce the mesh to at most ``target_faces`` triangles.

    Tries trimesh's built-in quadric decimation first; if it raises,
    falls back to pymeshlab with topology + boundary preservation.

    Args:
        mesh:         Input mesh.
        target_faces: Maximum triangle count after decimation.

    Returns:
        Decimated trimesh.Trimesh.
    """
    current = len(mesh.faces)
    if current <= target_faces:
        logger.debug(
            "Decimation skipped — mesh already has %d faces (<= %d target).",
            current, target_faces,
        )
        return mesh

    logger.info("Decimating mesh: %d → %d faces.", current, target_faces)

    # --- primary path: trimesh built-in ------------------------------------
    try:
        decimated = mesh.simplify_quadric_decimation(target_faces)
        logger.debug("Decimated with trimesh (result: %d faces).", len(decimated.faces))
        return decimated
    except Exception as exc:
        logger.warning("trimesh decimation failed (%s). Trying pymeshlab fallback.", exc)

    # --- fallback: pymeshlab -----------------------------------------------
    try:
        import pymeshlab  # type: ignore[import]

        ms = pymeshlab.MeshSet()
        ms.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
                face_matrix=np.asarray(mesh.faces, dtype=np.int32),
            )
        )
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preservetopology=True,
            preserveboundary=True,
        )
        current_mesh = ms.current_mesh()
        result = trimesh.Trimesh(
            vertices=current_mesh.vertex_matrix(),
            faces=current_mesh.face_matrix(),
            process=False,
        )
        logger.debug("Decimated with pymeshlab (result: %d faces).", len(result.faces))
        return result
    except ImportError:
        logger.warning("pymeshlab not installed — decimation skipped.")
    except Exception as exc:
        logger.warning("pymeshlab decimation failed (%s) — skipping decimation.", exc)

    return mesh


# ---------------------------------------------------------------------------
# Step E: Full postprocessing pipeline
# ---------------------------------------------------------------------------

def postprocess_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int = 200_000,
    normalize: bool = True,
) -> trimesh.Trimesh:
    """
    Full mesh cleanup pipeline.

    Steps (in order):
        1. Remove vertices with infinite / NaN values
        2. Remove unreferenced vertices and duplicate / degenerate faces
        3. Merge coincident vertices
        4. Keep largest connected component (remove floaters)
        5. Decimate to target_faces if needed
        6. Normalize to unit bounding box (if normalize=True)
        7. Final pass: remove unreferenced vertices after decimation

    Args:
        mesh:         Raw mesh from the shape pipeline.
        target_faces: Target triangle count for decimation.
        normalize:    If True, center at origin and scale to unit bounding box.

    Returns:
        Cleaned and normalized trimesh.Trimesh.
    """
    if isinstance(mesh, trimesh.Scene):
        logger.debug("postprocess_mesh: received Scene, concatenating.")
        mesh = mesh.dump(concatenate=True)

    mesh = mesh.copy()

    n_verts_before = len(mesh.vertices)
    n_faces_before = len(mesh.faces)
    logger.info(
        "Postprocessing mesh: %d vertices, %d faces.",
        n_verts_before, n_faces_before,
    )

    # 1-3: basic geometry cleanup
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.merge_vertices()

    # 4: floater removal
    mesh = keep_largest_component(mesh)

    # 5: decimation
    mesh = decimate_mesh(mesh, target_faces)

    # 6: normalize
    if normalize:
        mesh = normalize_mesh(mesh)

    # 7: final cleanup pass after all modifications
    mesh.remove_unreferenced_vertices()

    logger.info(
        "Postprocessing complete: %d → %d vertices, %d → %d faces.",
        n_verts_before, len(mesh.vertices),
        n_faces_before, len(mesh.faces),
    )
    return mesh


# ---------------------------------------------------------------------------
# Convenience: save mesh to disk
# ---------------------------------------------------------------------------

def save_mesh(mesh: trimesh.Trimesh, path, *, mkdir: bool = True) -> None:
    """
    Export a mesh to disk.

    Format is inferred from the file extension (.glb, .obj, .ply, etc.).

    Args:
        mesh:  Mesh to export.
        path:  Output file path.
        mkdir: Create parent directories if they don't exist.
    """
    from pathlib import Path
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    logger.info("Mesh saved: %s  (%d vertices, %d faces)", path, len(mesh.vertices), len(mesh.faces))
