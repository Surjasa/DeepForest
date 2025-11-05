"""Utilities to export detection crops and serve a minimal local gallery.

This module implements a small, local gallery MVP used by the CLI and
tests. It provides helpers to export thumbnails from a results DataFrame,
write a single-file ``index.html`` viewer, and serve the generated
directory with a tiny HTTP server for quick inspection.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image


def _resolve_image_path(row: Any, root_dir: str | None = None) -> str | None:
    """Resolve an image path from a result row.

    The row may be a mapping (dict-like) or a pandas Series with an
    ``image_path`` entry. If a ``root_dir`` is provided and the path is
    relative, the function returns an absolute path joined with the
    root.
    """

    image_path: str | None = None
    if isinstance(row, dict):
        image_path = row.get("image_path")
    else:
        image_path = (
            row.get("image_path") if "image_path" in getattr(row, "index", []) else None
        )

    if image_path and root_dir:
        p = Path(image_path)
        if not p.is_absolute():
            image_path = str(Path(root_dir) / image_path)

    return image_path


def _sanitize_label(label: str | None) -> str:
    """Create a filesystem-safe, short label from a value."""
    if label is None:
        return "unknown"
    s = str(label).strip().lower()
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return s or "unknown"


def export_to_gallery(
    results_df: Any,
    savedir: str,
    root_dir: str | None = None,
    padding: int = 8,
    thumb_size: tuple[int, int] = (256, 256),
    max_crops: int | None = None,
    sample_seed: int | None = None,
    point_size: int = 64,
    logger: logging.Logger | None = None,
    sample_by_image: bool = False,
    per_image_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Export crops from a results DataFrame into a static gallery.

    Args:
        results_df: pandas DataFrame with prediction rows. Rows must include either
            xmin,ymin,xmax,ymax columns or a `geometry` with `.bounds`. Each row
            should include `image_path` (absolute or relative) or `results_df.root_dir`
            can be set and will be used to resolve relative paths.
        savedir: directory to write thumbnails and metadata into.
        root_dir: optional root to resolve relative image paths.
        padding: pixels to pad around bounding boxes.
        thumb_size: (width, height) of output thumbnails.
        max_crops: maximum number of crops to export (sampling applied if fewer rows exist).
        sample_seed: deterministic seed for sampling when max_crops < available crops.
        point_size: box size (pixels) for point annotations.

    Writes:
        savedir/thumbnails/*.jpg
        savedir/metadata.json
        savedir/metadata.csv

    Returns:
        metadata (list of dict)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    savedir_p = Path(savedir)
    savedir_p.mkdir(parents=True, exist_ok=True)
    thumbs_dir = savedir_p / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)

    # Build list of candidate indices that have image_path and bbox info
    # Optionally sample by image to ensure broad coverage across images.
    img_to_indices: dict[str, list[int]] = {}
    for i, row in results_df.iterrows():
        img_path = _resolve_image_path(
            row, root_dir or getattr(results_df, "root_dir", None)
        )
        if not img_path:
            continue
        img_to_indices.setdefault(img_path, []).append(i)

    # Prepare sampling
    image_paths = list(img_to_indices.keys())
    if sample_seed is not None:
        rng = np.random.RandomState(sample_seed)
        rng.shuffle(image_paths)

    indices: list[int] = []
    if sample_by_image:
        # If per_image_limit is provided, take up to that many rows per image (fast path).
        if per_image_limit is not None:
            for img in image_paths:
                rows = img_to_indices[img][:]
                if sample_seed is not None:
                    rng.shuffle(rows)
                selected = rows[:per_image_limit]
                indices.extend(selected)
                if max_crops is not None and len(indices) >= max_crops:
                    indices = indices[:max_crops]
                    break
        else:
            # Round-robin across images to distribute crops evenly.
            # Prepare per-image shuffled queues
            queues = []
            for img in image_paths:
                rows = img_to_indices[img][:]
                if sample_seed is not None:
                    rng.shuffle(rows)
                queues.append(list(rows))

            # Round-robin selection
            more = True
            while more and (max_crops is None or len(indices) < max_crops):
                more = False
                for q in queues:
                    if not q:
                        continue
                    more = True
                    indices.append(q.pop(0))
                    if max_crops is not None and len(indices) >= max_crops:
                        break
    else:
        # flatten per-row sampling
        for img in image_paths:
            rows = img_to_indices[img]
            indices.extend(rows)
        if sample_seed is not None:
            rng.shuffle(indices)
        if max_crops is not None:
            indices = indices[:max_crops]

    metadata: list[dict[str, Any]] = []
    written = 0
    for idx in indices:
        row = results_df.loc[idx]
        img_path = _resolve_image_path(
            row, root_dir or getattr(results_df, "root_dir", None)
        )
        if not img_path or not Path(img_path).exists():
            logger.warning("Image not found, skipping: %s", img_path)
            continue

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                # extract bbox
                xmin = ymin = xmax = ymax = None
                for names in [
                    ("xmin", "ymin", "xmax", "ymax"),
                    ("x_min", "y_min", "x_max", "y_max"),
                ]:
                    if all(n in row.index for n in names):
                        xmin, ymin, xmax, ymax = (float(row[names[i]]) for i in range(4))
                        break

                # geometry bounds fallback
                if xmin is None and "geometry" in row.index:
                    geom = row["geometry"]
                    if hasattr(geom, "bounds"):
                        b = geom.bounds
                        xmin, ymin, xmax, ymax = (
                            float(b[0]),
                            float(b[1]),
                            float(b[2]),
                            float(b[3]),
                        )
                    elif hasattr(geom, "x") and hasattr(geom, "y"):
                        # point geometry
                        cx, cy = float(geom.x), float(geom.y)
                        half = point_size / 2.0
                        xmin, ymin, xmax, ymax = (
                            cx - half,
                            cy - half,
                            cx + half,
                            cy + half,
                        )

                # fallback for point columns
                if xmin is None and all(k in row.index for k in ("x", "y")):
                    cx, cy = float(row["x"]), float(row["y"])
                    half = point_size / 2.0
                    xmin, ymin, xmax, ymax = cx - half, cy - half, cx + half, cy + half

                if xmin is None:
                    logger.warning("No bounding box info for row %s, skipping", idx)
                    continue

                # apply padding and clip
                xmin = max(int(math.floor(xmin - padding)), 0)
                ymin = max(int(math.floor(ymin - padding)), 0)
                xmax = min(int(math.ceil(xmax + padding)), im.width)
                ymax = min(int(math.ceil(ymax + padding)), im.height)

                if xmax <= xmin or ymax <= ymin:
                    logger.warning("Empty crop for row %s, skipping", idx)
                    continue

                crop = im.crop((xmin, ymin, xmax, ymax))
                # resize to fixed thumb size
                crop = crop.resize((thumb_size[0], thumb_size[1]), resample=Image.LANCZOS)

                label = None
                if "label" in getattr(row, "index", []):
                    label = row.get("label")
                label_safe = _sanitize_label(label)

                unique = uuid4().hex[:8]
                fname = f"{written:06d}_{label_safe}_{unique}.jpg"
                out_path = thumbs_dir / fname
                crop.save(str(out_path), quality=90)

                meta = {
                    "crop_id": written,
                    "filename": str(Path("thumbnails") / fname),
                    "source_image": img_path,
                    "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "label": label if label is not None else "Unknown",
                    "score": float(row.get("score"))
                    if (
                        "score" in getattr(row, "index", [])
                        and row.get("score") is not None
                    )
                    else None,
                    "width": im.width,
                    "height": im.height,
                }
                metadata.append(meta)
                written += 1
        except Exception as e:
            logger.warning("Error processing %s: %s", img_path, e)
            continue

    # write metadata files
    with open(savedir_p / "metadata.json", "w", encoding="utf8") as fh:
        json.dump(metadata, fh, indent=2)

    # write CSV for convenience
    if metadata:
        csv_path = savedir_p / "metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf8") as csvfile:
            writer = csv.writer(csvfile)
            header = [
                "crop_id",
                "filename",
                "source_image",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "label",
                "score",
                "width",
                "height",
            ]
            writer.writerow(header)
            for m in metadata:
                xmin, ymin, xmax, ymax = m["bbox"]
                writer.writerow(
                    [
                        m["crop_id"],
                        m["filename"],
                        m["source_image"],
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        m.get("label"),
                        m.get("score"),
                        m.get("width"),
                        m.get("height"),
                    ]
                )

    return metadata


def write_gallery_html(
    savedir: str, title: str = "DeepForest Gallery", page_size: int = 200
) -> None:
    """Write a minimal single-file index.html that displays the thumbnails and
    metadata.

    page_size: number of thumbnails to render per "Load more" click. This
    reduces initial DOM operations for large galleries (the full metadata is
    still fetched, but rendering is incremental).
    """
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 1rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(160px,1fr)); grid-gap: 8px; }}
    .card {{ border: 1px solid #ddd; padding: 4px; border-radius:4px; background:#fff }}
    .card img {{ width:100%; height:auto; display:block }}
    .meta {{ font-size: 12px; color: #333 }}
    .filters {{ margin-bottom: 12px }}
  </style>
  </head>
<body>
  <h1>{title}</h1>
  <div class="filters">
    <label>Label filter: <input id="labelFilter" placeholder="label substring"></label>
  </div>
    <div id="grid" class="grid"></div>
    <div style="text-align:center; margin-top:8px;"><button id="loadMore">Load more</button></div>
    <script>
    async function load() {{
        const res = await fetch('metadata.json');
        const data = await res.json();
        window._galleryData = data;
        window._pageIndex = 0;
        window._pageSize = {page_size};
        renderPage();
    }}
    function renderItem(item) {{
        const card = document.createElement('div'); card.className = 'card';
        const img = document.createElement('img'); img.src = item.filename; card.appendChild(img);
        const meta = document.createElement('div'); meta.className='meta';
        meta.textContent = `${{item.label}} (id:${{item.crop_id}}) score:${{item.score}}`;
        card.appendChild(meta);
        return card;
    }}
    function renderPage() {{
        const data = window._galleryData || [];
        const grid = document.getElementById('grid');
        const filter = document.getElementById('labelFilter').value.toLowerCase();
        const start = window._pageIndex * window._pageSize;
        const end = Math.min(start + window._pageSize, data.length);
        for (let i = start; i < end; i++) {{
            const item = data[i];
            if (filter && (!item.label || item.label.toLowerCase().indexOf(filter) === -1)) continue;
            grid.appendChild(renderItem(item));
        }}
        window._pageIndex += 1;
        if (window._pageIndex * window._pageSize >= data.length) {{
            document.getElementById('loadMore').style.display = 'none';
        }} else {{
            document.getElementById('loadMore').style.display = 'inline-block';
        }}
    }}
    document.getElementById('labelFilter').addEventListener('input', () => {{
        const grid = document.getElementById('grid');
        grid.innerHTML = '';
        window._pageIndex = 0;
        renderPage();
    }});
    document.getElementById('loadMore').addEventListener('click', () => renderPage());
    load();
    </script>
</body>
</html>"""

    with open(Path(savedir) / "index.html", "w", encoding="utf8") as fh:
        fh.write(html)


def start_local_gallery(
    savedir: str, host: str = "127.0.0.1", port: int = 0, open_browser: bool = True
) -> TCPServer:
    """Serve the savedir with a tiny HTTP server in a background thread.

    Returns the server object (call .shutdown() to stop).
    """
    handler = SimpleHTTPRequestHandler

    # Use threaded server to avoid blocking
    class _TCPServer(TCPServer):
        allow_reuse_address = True

    # change dir for handler
    cwd = Path.cwd()
    try:
        Path(savedir).resolve()
        # change dir for handler
        import os as _os

        _os.chdir(savedir)
        httpd = _TCPServer((host, port), handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        url = f"http://{host}:{port}/index.html"
        if open_browser:
            webbrowser.open(url)
        return httpd
    finally:
        # restore previous working directory
        import os as _os

        _os.chdir(cwd)
