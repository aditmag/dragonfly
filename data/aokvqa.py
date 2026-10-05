"""A-OKVQA multiple choice (exact labels) -> out/aokvqa.jsonl. The test split has no labels, so train + val only."""
from collections import defaultdict

from common import RAW, coco_id, family, load, write_by_image


def main():
    by_image = defaultdict(list)
    for split in ("train", "val"):
        for r in load(RAW / f"aokvqa/aokvqa_v1p0_{split}.json"):
            opts = r["choices"]
            by_image[coco_id(r["image_id"])].append({
                "type": "choice", "family": family(r["question"]), "source": "aokvqa", "ref": r["question_id"],
                "q": r["question"], "options": opts,
                "target": {o: float(i == r["correct_choice_idx"]) for i, o in enumerate(opts)},
                "na": 0.0, "label_kind": "exact",
            })
    write_by_image("aokvqa", by_image)


if __name__ == "__main__":
    main()
