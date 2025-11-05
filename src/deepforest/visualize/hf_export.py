"""Hugging Face dataset export helpers for DeepForest gallery output.

This module provides a lightweight exporter that converts a gallery
directory (thumbnails + metadata.json) into a Hugging Face-friendly
dataset folder structure (images + metadata.jsonl). The implementation
avoids importing the `datasets` library so it can run in CI without
extra dependencies. If users want to create a `datasets.Dataset` or
push to the Hub, they can do that from the produced folder or enable
optional dependencies.

Functions:
  export_to_huggingface(savedir, out_dir=None, dataset_name=None, push_to_hub=False, token=None)

The exporter creates:
  out_dir/images/...  (copied thumbnails)
  out_dir/metadata.jsonl  (one JSON object per line with fields image,label,bbox,score,source_image)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any


def _read_metadata(savedir: Path) -> list[dict[str, Any]]:
    meta_path = savedir / "metadata.json"
    if not meta_path.exists():
        raise ValueError(f"metadata.json not found in {savedir}")
    with meta_path.open("r", encoding="utf8") as fh:
        return json.load(fh)


def export_to_huggingface(
    savedir: str | Path,
    out_dir: str | Path | None = None,
    dataset_name: str | None = None,
    push_to_hub: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    """Export a gallery directory into a Hugging Face-compatible folder.

    This writes a simple folder with `images/` and `metadata.jsonl` which
    can be consumed by `datasets.Image()` or loaded with `datasets` as a
    Dataset. No `datasets` import is required for this step.

    Args:
        savedir: Path to gallery output (contains thumbnails/ and metadata.json).
        out_dir: Destination folder for HF dataset artifacts. Defaults to
            ``savedir/hf_dataset``.
        dataset_name: Optional dataset name (used only when pushing to hub).
        push_to_hub: If True, attempts to create and push a HF Dataset. This
            requires `datasets` and `huggingface_hub` to be installed and
            authenticated; otherwise an error is raised.
        token: Optional HF token to authenticate when pushing.

    Returns:
        A dict with keys: out_dir (Path), records_written (int), push_info (optional)
    """
    logger = logging.getLogger(__name__)
    savedir_p = Path(savedir)
    if out_dir is None:
        out_dir_p = savedir_p / "hf_dataset"
    else:
        out_dir_p = Path(out_dir)

    images_src = savedir_p / "thumbnails"
    if not images_src.exists():
        raise ValueError(f"Thumbnails directory not found in {savedir_p}")

    out_images = out_dir_p / "images"
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    metadata = _read_metadata(savedir_p)

    # Copy images and write JSONL records
    jsonl_path = out_dir_p / "metadata.jsonl"
    written = 0
    with jsonl_path.open("w", encoding="utf8") as outfh:
        for rec in metadata:
            src_fname = rec.get("filename")
            if not src_fname:
                logger.warning("Skipping record without filename: %s", rec)
                continue
            src_path = savedir_p / src_fname
            if not src_path.exists():
                logger.warning("Image missing, skipping: %s", src_path)
                continue

            dst_fname = Path(src_fname).name
            dst_path = out_images / dst_fname
            shutil.copy2(src_path, dst_path)

            out_record = {
                "image": str(Path("images") / dst_fname),
                "label": rec.get("label"),
                "bbox": rec.get("bbox"),
                "score": rec.get("score"),
                "source_image": rec.get("source_image"),
            }
            outfh.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            written += 1

    result: dict[str, Any] = {"out_dir": str(out_dir_p), "records_written": written}

    if push_to_hub:
        try:
            # optional import to push; keep dependency optional
            from datasets import load_dataset
            from huggingface_hub import HfApi
        except Exception as e:
            raise RuntimeError(
                "Pushing to hub requires `datasets` and `huggingface_hub`."
            ) from e

        # Create a Dataset from the JSONL we just wrote
        ds = load_dataset("json", data_files={"train": str(jsonl_path)})["train"]

        api = HfApi()
        repo_id = dataset_name or "deepforest/gallery-dataset"
        if token:
            api.create_repo(token=token, name=repo_id, exist_ok=True)
        else:
            api.create_repo(name=repo_id, exist_ok=True)

        ds.push_to_hub(repo_id, token=token)
        result["push_info"] = {"repo_id": repo_id}

    return result
