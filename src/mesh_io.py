"""
Mesh I/O utilities for the Hunyuan3D pipeline.

MeshSaver — saves untextured trimesh meshes in the pipeline's configured
output format (GLB or OBJ).  Used for intermediate saves (raw mesh,
postprocessed, UV-unwrapped) so every file on disk uses the same extension
selected in the UI.

The final textured export is handled separately by src/export_glb.py, which
also manages the OBJ-to-GLB conversion and PBR material embedding.
"""

from __future__ import annotations

import logging
from pathlib import Path

import trimesh

logger = logging.getLogger(__name__)


class MeshSaver:
    """
    Saves untextured trimesh meshes using the pipeline's output format.

    The format is set once at construction time (from the UI dropdown or a
    CLI flag) so all intermediate saves are consistent.

    Supported formats: "glb", "obj".
    trimesh infers the actual writer from the file suffix, so both formats
    are handled by trimesh's built-in exporters without extra dependencies.

    Example:
        saver = MeshSaver("obj")
        path = saver.save(raw_mesh, save_dir, "mesh_raw")
        # → save_dir/mesh_raw.obj
    """

    SUPPORTED: frozenset[str] = frozenset({"glb", "obj"})

    def __init__(self, fmt: str = "glb") -> None:
        """
        Args:
            fmt: Output format, must be "glb" or "obj" (case-insensitive).
        """
        fmt = fmt.lower()
        if fmt not in self.SUPPORTED:
            raise ValueError(
                f"MeshSaver: fmt must be one of {sorted(self.SUPPORTED)}, got {fmt!r}"
            )
        self.fmt = fmt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def path(self, directory: Path | str, stem: str) -> Path:
        """
        Return the expected output path without writing anything.

        Args:
            directory: Target directory.
            stem:      Filename without extension, e.g. "mesh_raw".

        Returns:
            Path: directory/stem.{fmt}
        """
        return Path(directory) / f"{stem}.{self.fmt}"

    def save(
        self,
        mesh: trimesh.Trimesh,
        directory: Path | str,
        stem: str,
    ) -> Path:
        """
        Export a trimesh to directory/stem.{fmt}.

        Creates the directory if it does not exist.

        Args:
            mesh:      The trimesh.Trimesh to export.
            directory: Target directory.
            stem:      Filename without extension.

        Returns:
            Path: the file that was written.
        """
        out = self.path(directory, stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out))
        logger.debug("Saved mesh to %s (%d verts, %d faces).",
                     out, len(mesh.vertices), len(mesh.faces))
        return out
