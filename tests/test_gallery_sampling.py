from pathlib import Path

import pandas as pd
from PIL import Image

from deepforest.visualize import export_to_gallery


def _make_image(tmp_path: Path, name: str):
    p = tmp_path / name
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img.save(p)
    return str(p)


def test_sample_by_image_round_robin(tmp_path):
    # Create 3 images, each with 4 detections (total 12)
    img_paths = [
        _make_image(tmp_path, f"img{i}.png") for i in range(3)
    ]

    rows = []
    for i, img in enumerate(img_paths):
        for j in range(4):
            rows.append({
                "image_path": Path(img).name,
                "xmin": 1,
                "ymin": 1,
                "xmax": 10,
                "ymax": 10,
                "label": f"L{i}",
                "score": 0.5,
            })

    df = pd.DataFrame(rows)
    df.root_dir = str(tmp_path)

    out = tmp_path / "g"
    # request 6 crops, expect round-robin distribution: balanced across images
    meta = export_to_gallery(df, str(out), max_crops=6, sample_by_image=True, sample_seed=42)
    assert len(meta) == 6

    # count crops per image
    counts: dict[str, int] = {}
    for m in meta:
        counts[m["source_image"]] = counts.get(m["source_image"], 0) + 1

    # Each image should be represented and distribution should be roughly even
    assert len(counts) == 3
    vals = sorted(counts.values())
    # difference between max and min should be <= 1 for round-robin
    assert vals[-1] - vals[0] <= 1
