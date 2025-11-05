import json
from pathlib import Path

from PIL import Image

from deepforest.visualize import export_to_gallery
from deepforest.visualize.hf_export import export_to_huggingface


def test_export_to_huggingface(tmp_path: Path):
    # create a small test image
    img_path = tmp_path / "img1.png"
    Image.new("RGB", (64, 64), color=(100, 120, 140)).save(img_path)

    # build dataframe-like input via pandas DataFrame
    import pandas as pd

    df = pd.DataFrame([
        {"image_path": img_path.name, "xmin": 1, "ymin": 1, "xmax": 20, "ymax": 20, "label": "Tree", "score": 0.9}
    ])
    df.root_dir = str(tmp_path)

    gallery_dir = tmp_path / "gallery"
    meta = export_to_gallery(df, str(gallery_dir), root_dir=None, max_crops=10)

    out = tmp_path / "hf"
    res = export_to_huggingface(gallery_dir, out_dir=out)

    assert Path(res["out_dir"]).exists()
    metadata_jsonl = out / "metadata.jsonl"
    assert metadata_jsonl.exists()
    lines = list(metadata_jsonl.read_text(encoding="utf8").strip().splitlines())
    assert len(lines) == len(meta)
    # verify images copied
    images_dir = out / "images"
    assert images_dir.exists()
    assert any(images_dir.iterdir())
    # verify JSON content keys
    rec = json.loads(lines[0])
    assert "image" in rec and "label" in rec and "bbox" in rec
