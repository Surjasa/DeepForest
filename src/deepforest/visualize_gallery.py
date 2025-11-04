"""Compatibility shim for the gallery utilities.

Exports: export_to_gallery, write_gallery_html, start_local_gallery
which are implemented in ``deepforest.visualize.gallery``. This shim
keeps old import paths working while the real implementation lives in
the ``deepforest.visualize`` package.
"""

try:
    from deepforest.visualize.gallery import (
        export_to_gallery,  # noqa: F401
        start_local_gallery,  # noqa: F401
        write_gallery_html,  # noqa: F401
    )
except Exception:  # pragma: no cover - safe fallback when package not available

    def export_to_gallery(*args, **kwargs):
        raise RuntimeError("deepforest.visualize.gallery is not available")

    def write_gallery_html(*args, **kwargs):
        raise RuntimeError("deepforest.visualize.gallery is not available")

    def start_local_gallery(*args, **kwargs):
        raise RuntimeError("deepforest.visualize.gallery is not available")
