
# Export detections to a local gallery

This page explains how to export detection crops (thumbnails) and metadata into a small, static local gallery so you can quickly inspect many predictions.

## export_to_gallery

### Description

`export_to_gallery` extracts crops from detection results, writes thumbnails to disk, and saves per-crop metadata (CSV and JSON) suitable for a simple static viewer.

### Arguments

- `results_df` (pandas.DataFrame): DataFrame containing detection results. Required columns: `image_path`, `xmin`, `ymin`, `xmax`, `ymax`. Optional: `label`, `score`.
- `savedir` (str): Directory to write thumbnails and metadata. Will be created if missing.
- `root_dir` (str, optional): Base directory to resolve relative `image_path` values.
- `max_crops` (int, optional): Maximum number of crops to export. Default: `None` (export all).
- `sample_by_image` (bool, optional): If true, sample images first then crops per image to ensure broad coverage. Default: `False`.
- `per_image_limit` (int, optional): Maximum crops per source image when `sample_by_image=True`.
- `sample_seed` (int, optional): RNG seed for deterministic sampling.
- `thumb_size` (int or tuple, optional): Thumbnail size in pixels (e.g. `128` or `(128,128)`).

### Returns

- `dict`: Summary information including number of thumbnails written and paths to `metadata.csv`/`metadata.json`.

### Example (Python)

```python
from deepforest.visualize.gallery import export_to_gallery, write_gallery_html, start_local_gallery

# results_df is a pandas DataFrame with columns: image_path, xmin, ymin, xmax, ymax, (optional) label, score
out = export_to_gallery(
	results_df,
	savedir="/tmp/df_gallery",
	root_dir="/data/images",
	max_crops=500,
	sample_by_image=True,
	per_image_limit=10,
	sample_seed=42,
)

write_gallery_html(out["savedir"])  # writes a simple index.html viewer
# serve locally (optional) — returns a Server object
server = start_local_gallery(out["savedir"])  # call server.shutdown() to stop
```

### Example (CLI)

```powershell
# export crops from CSV of results
deepforest gallery export results.csv -s out/gallery --root-dir C:\data\images --max-crops 500 --sample-by-image --open
```

## Notes

- Thumbnails are written under `thumbnails/` inside `savedir` and metadata is written to `metadata.csv` and `metadata.json`.
- Use `sample_by_image=True` with `per_image_limit` to prefer coverage across many images when `max_crops` is limited.
- The exported static viewer is intentionally minimal — follow-up work can add Hugging Face Dataset export or Renumics Spotlight ingestion.

## See also

- `deepforest.visualize` helpers and `deepforest` CLI.
