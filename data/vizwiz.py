"""VizWiz unanswerable yes/no-phrased questions -> bool with na = 1 (target null) -> out/vizwiz.jsonl + out/images_vizwiz.jsonl.

Most VizWiz questions are open-ended ("What is this?") and can't be typed, so only plain yes/no phrasing is kept.
Images only ship as zips (no per-image URL); member is matched by basename.
"""
import re
from collections import defaultdict

from common import OUT, RAW, load, write_by_image, write_jsonl

BOOL = re.compile(r"^\s*(is|are|was|were|does|do|did|has|have|had)\b", re.I)
OPEN = re.compile(r"\b(or|what|which|who|where|when|how|why|tell)\b", re.I)


def main():
    by_image, images = defaultdict(list), []
    for split in ("train", "val"):
        for r in load(RAW / f"vizwiz/{split}.json"):
            q = r["question"].strip()
            if r["answerable"] or not BOOL.match(q) or OPEN.search(q) or q.count("?") > 1:
                continue
            image_id = "vizwiz_" + re.sub(r"^VizWiz_|\.jpg$", "", r["image"])
            by_image[image_id].append({
                "type": "bool", "family": "unanswerable", "source": "vizwiz", "ref": r["image"],
                "q": q, "target": None, "na": 1.0, "label_kind": "exact",
            })
            images.append({"image_id": image_id, "source": "vizwiz", "width": None, "height": None, "license": "CC BY 4.0",
                           "archive": f"https://vizwiz.cs.colorado.edu/VizWiz_final/images/{split}.zip", "member": r["image"]})
    write_by_image("vizwiz", by_image)
    write_jsonl(OUT / "images_vizwiz.jsonl", images)


if __name__ == "__main__":
    main()
