import json
from pathlib import Path

import pandas as pd
from PIL import Image

from deepforest.visualize import export_to_gallery, write_gallery_html


def test_export_to_gallery(tmp_path):
    # create a small test image
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (128, 128), color=(123, 222, 64))
    img.save(img_path)

    # build dataframe
    df = pd.DataFrame([
        {"image_path": str(img_path.name), "xmin": 10, "ymin": 10, "xmax": 80, "ymax": 80, "label": "Tree", "score": 0.9}
    ])
    # set root_dir so relative path resolves
    df.root_dir = str(tmp_path)

    outdir = tmp_path / "gallery"
    metadata = export_to_gallery(df, str(outdir), root_dir=None, max_crops=10)

    # basic assertions
    meta_json = outdir / "metadata.json"
    assert meta_json.exists()
    with meta_json.open() as fh:
        data = json.load(fh)
    assert isinstance(data, list)
    assert len(data) == len(metadata) == 1

    thumbs = outdir / "thumbnails"
    assert thumbs.is_dir()
    files = list(thumbs.iterdir())
    assert len(files) == 1

    # write simple html and check contents
    write_gallery_html(str(outdir))
    index = outdir / "index.html"
    assert index.exists()
    content = index.read_text(encoding="utf8")
    assert "DeepForest Gallery" in content or "DeepForest" in content
