"""POPE (evaluation only) -> out/pope.jsonl. merge.py forces these images into the test split."""
import re
from collections import defaultdict

from common import OUT, RAW, coco_id, read_jsonl, write_jsonl


def main():
    by_image = defaultdict(list)
    for split in ("random", "popular", "adversarial"):
        for r in read_jsonl(RAW / f"pope/coco_pope_{split}.json"):
            image_id = coco_id(re.search(r"(\d+)\.jpg$", r["image"]).group(1))
            yes = 1.0 if r["label"] == "yes" else 0.0
            by_image[image_id].append({
                "type": "bool",
                "family": "presence",
                "source": f"pope_{split}",
                "ref": r["question_id"],
                "q": r["text"],
                "target": {"yes": yes, "no": 1.0 - yes},
                "na": 0.0,
                "label_kind": "exact",
            })
    write_jsonl(OUT / "pope.jsonl", ({"image_id": k, "questions": v} for k, v in by_image.items()))


if __name__ == "__main__":
    main()
