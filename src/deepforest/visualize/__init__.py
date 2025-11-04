"""Visualization subpackage for DeepForest.

This package exposes visualization helpers. The gallery utilities live in
``deepforest.visualize.gallery`` and are re-exported here for convenience.

We also attempt to dynamically load the legacy top-level module
``visualize.py`` (if present) and re-export its public names so older
imports like ``from deepforest.visualize import plot_results`` continue to work.
"""

import importlib.util
import logging
import os

from .gallery import export_to_gallery, start_local_gallery, write_gallery_html

__all__: list[str] = [
    "export_to_gallery",
    "write_gallery_html",
    "start_local_gallery",
]

# Attempt to load the legacy top-level module `visualize.py` (from before this
# subpackage existed) so existing imports like `from deepforest.visualize import ...`
# continue to work. If the legacy file exists we import it under a private
# module name and re-export its public, non-underscored names.
_here = os.path.dirname(__file__)
_legacy_path = os.path.normpath(os.path.join(_here, "..", "visualize.py"))
_legacy_module_name = "deepforest.visualize_legacy"

if os.path.exists(_legacy_path):
    spec = importlib.util.spec_from_file_location(_legacy_module_name, _legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    try:
        # Execute the legacy module in its own module object
        spec.loader.exec_module(legacy)  # type: ignore[attr-defined]
        for _name in dir(legacy):
            if _name.startswith("_"):
                continue
            if _name in globals():
                # Don't overwrite existing names (like our gallery exports)
                continue
            globals()[_name] = getattr(legacy, _name)
            __all__.append(_name)
    except Exception as e:  # pragma: no cover - best-effort import
        logging.warning(f"Failed to load legacy visualize.py: {e}")
