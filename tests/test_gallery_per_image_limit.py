from pathlib import Path

import pandas as pd
from PIL import Image

from deepforest.visualize import export_to_gallery


def _make_image(tmp_path: Path, name: str):
    p = tmp_path / name
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img.save(p)
    return str(p)


def test_per_image_limit(tmp_path):
    # Create 2 images, each with 5 detections (total 10)
    imgs = [_make_image(tmp_path, f"a{i}.png") for i in range(2)]

    rows = []
    for i, img in enumerate(imgs):
        for j in range(5):
            rows.append({
                "image_path": Path(img).name,
                "xmin": 1,
                "ymin": 1,
                "xmax": 20,
                "ymax": 20,
                "label": f"lab{i}",
                "score": 0.7,
            })

    df = pd.DataFrame(rows)
    df.root_dir = str(tmp_path)

    out = tmp_path / "gallery2"
    # use per_image_limit=2 to ensure at most 2 crops per image
    meta = export_to_gallery(df, str(out), max_crops=10, sample_by_image=True, per_image_limit=2, sample_seed=1)

    counts = {}
    for m in meta:
        counts[m["source_image"]] = counts.get(m["source_image"], 0) + 1

    # each image should have at most 2 crops
    for v in counts.values():
        assert v <= 2
