Hugging Face dataset export
==========================

This page describes how to export a local DeepForest gallery into a
Hugging Face-compatible folder (images + JSONL metadata). The exporter
produces artifacts that can be loaded into `datasets` or pushed to the
Hugging Face Hub (optional).

Basic usage
-----------

1. Create a gallery using `export_to_gallery` or the CLI:

```
python -m deepforest.scripts.cli gallery export -i preds.csv -o ./out_gallery --root-dir .
```

2. Export to a HF-friendly folder:

```python
from deepforest.visualize.hf_export import export_to_huggingface

res = export_to_huggingface("out_gallery", out_dir="out_gallery/hf_dataset")
print(res)
```

The output folder will contain:

- `images/` — copied thumbnails
- `metadata.jsonl` — one JSON record per line with keys: `image`, `label`, `bbox`, `score`, `source_image`

Optional: push to the Hub
-------------------------

To push the dataset to the Hugging Face Hub pass `push_to_hub=True`. This
requires the `datasets` and `huggingface_hub` packages and an authenticated
session (or token). Pushing is optional and not required to use the exporter.

Example (optional):

```python
export_to_huggingface("out_gallery", out_dir="out_gallery/hf_dataset", push_to_hub=True, dataset_name="username/gallery-name")
```

Notes
-----
- The exporter intentionally does not import `datasets` at module import
  time so it remains testable in CI without extra dependencies.
- For very large galleries you may prefer to stream metadata or use a
  server-backed API to paginate metadata rather than shipping a huge
  single JSON file.
