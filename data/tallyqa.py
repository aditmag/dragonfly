"""TallyQA counting questions -> score 0-9 (exact) -> out/tallyqa.jsonl. Counts above 9 are dropped.

The source name carries TallyQA's own data_source (amt = human-written complex questions; the rest are
imported or templated), so image selection can prefer the human ones.
"""
from collections import defaultdict

from common import RAW, coco_id, load, vg_image, write_by_image


def image_of(path):  # 'train2014/COCO_train2014_000000247712.jpg' or 'VG_100K/4.jpg'
    folder, name = path.split("/")
    stem = name[:-4]
    return coco_id(stem.rsplit("_", 1)[1]) if folder in ("train2014", "val2014") else vg_image(stem)


def main():
    by_image, dropped = defaultdict(list), 0
    for split in ("train", "test"):
        for r in load(RAW / f"tallyqa/{split}.json"):
            image = image_of(r["image"])
            if not 0 <= r["answer"] <= 9 or image is None:
                dropped += 1
                continue
            by_image[image].append({
                "type": "score", "family": "count", "source": f"tallyqa_{r['data_source']}", "ref": r["question_id"],
                "q": r["question"], "scale": [0, 9], "anchors": ["none", "nine"],
                "target": [float(i == r["answer"]) for i in range(10)], "na": 0.0, "label_kind": "exact",
            })
    print(f"tallyqa: {sum(map(len, by_image.values())):,} kept, {dropped:,} dropped (count > 9 or unknown image)")
    write_by_image("tallyqa", by_image)


if __name__ == "__main__":
    main()
