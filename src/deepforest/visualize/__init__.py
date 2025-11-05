"""Visualization subpackage for DeepForest.

This package exposes visualization helpers. The gallery utilities live in
``deepforest.visualize.gallery`` and are re-exported here for convenience.

We provide an optional, opt-in loader for an older top-level ``visualize.py``
file (used by older releases). Loading that legacy module is disabled by
default; set the environment variable ``DEEPFOREST_LOAD_LEGACY_VISUALIZE=1``
to enable it. Making the loader opt-in avoids surprising import-time side
effects and keeps the package import fast and predictable.
"""

import importlib.util
import logging
import os
from collections.abc import Iterable

from .gallery import export_to_gallery, start_local_gallery, write_gallery_html

__all__: list[str] = ["export_to_gallery", "write_gallery_html", "start_local_gallery"]

# Optional legacy loader: only run when the environment variable is explicitly set.
# This avoids unexpected import-time behavior and follows the principle of least
# surprise for downstream consumers.
_enable_legacy = os.environ.get("DEEPFOREST_LOAD_LEGACY_VISUALIZE", "0") == "1"
if _enable_legacy:
    _here = os.path.dirname(__file__)
    _legacy_path = os.path.normpath(os.path.join(_here, "..", "visualize.py"))
    if os.path.exists(_legacy_path):
        spec = importlib.util.spec_from_file_location(
            "deepforest._visualize_legacy", _legacy_path
        )
        legacy = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(legacy)  # type: ignore[attr-defined]
            # Re-export public names from legacy module but avoid overwriting current names
            for _name in dir(legacy):
                if _name.startswith("_") or _name in globals():
                    continue
                globals()[_name] = getattr(legacy, _name)
                __all__.append(_name)
        except Exception as e:  # pragma: no cover - best-effort import
            logging.warning("Failed to load legacy visualize.py: %s", e)
