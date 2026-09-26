"""VQAv2 yes/no questions -> bool records in out/vqav2.jsonl.

The label is the share of "yes" among the human answers (a human spread, not a majority vote).
"""
from collections import defaultdict

from common import OUT, RAW, coco_id, family, load, write_jsonl

MIN_YES_NO = 8  # skip questions where fewer than 8 of the 10 answers are a plain yes or no


def main():
    by_image = defaultdict(list)
    kept = skipped = 0
    for split in ("train2014", "val2014"):
        questions = {q["question_id"]: q["question"]
                     for q in load(RAW / f"vqav2/v2_OpenEnded_mscoco_{split}_questions.json")["questions"]}
        for a in load(RAW / f"vqav2/v2_mscoco_{split}_annotations.json")["annotations"]:
            if a["answer_type"] != "yes/no":
                continue
            answers = [x["answer"].strip().lower() for x in a["answers"]]
            yes, no = answers.count("yes"), answers.count("no")
            if yes + no < MIN_YES_NO:
                skipped += 1
                continue
            q = questions[a["question_id"]]
            p_yes = round(yes / (yes + no), 4)
            by_image[coco_id(a["image_id"])].append({
                "type": "bool",
                "family": family(q),
                "source": "vqav2",
                "ref": a["question_id"],
                "q": q,
                "target": {"yes": p_yes, "no": round(1 - p_yes, 4)},
                "na": 0.0,
                "label_kind": "human_soft",
            })
            kept += 1
    print(f"vqav2 yes/no: kept {kept:,}, skipped {skipped:,} with <{MIN_YES_NO} yes/no answers")
    write_jsonl(OUT / "vqav2.jsonl", ({"image_id": k, "questions": v} for k, v in by_image.items()))


if __name__ == "__main__":
    main()
