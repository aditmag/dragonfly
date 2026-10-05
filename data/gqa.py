"""GQA verification questions (yes/no, exact labels) -> out/gqa.jsonl. Balanced train + val."""
from collections import defaultdict

from common import RAW, family, load, vg_image, write_by_image


def main():
    by_image, skipped = defaultdict(list), 0
    for split in ("train", "val"):
        for qid, r in load(RAW / f"gqa/{split}_balanced_questions.json").items():
            if r["types"]["structural"] != "verify" or r["answer"] not in ("yes", "no"):
                continue
            image = vg_image(r["imageId"])
            if image is None:
                skipped += 1
                continue
            yes = float(r["answer"] == "yes")
            by_image[image].append({
                "type": "bool", "family": family(r["question"]), "source": "gqa", "ref": qid,
                "q": r["question"], "target": {"yes": yes, "no": 1 - yes}, "na": 0.0, "label_kind": "exact",
            })
    print(f"gqa verify: {sum(map(len, by_image.values())):,} questions, {skipped:,} skipped (image not in VG)")
    write_by_image("gqa", by_image)


if __name__ == "__main__":
    main()
