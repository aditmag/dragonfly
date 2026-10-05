"""KonIQ-10k image quality -> score 1-5 with the 5-vote histogram as target -> out/koniq.jsonl + out/images_koniq.jsonl.

Images only ship as zips; we use the 512x384 one (767 MB), which fits the 448 px model budget.
"""
import csv
import hashlib

from common import OUT, RAW, write_by_image, write_jsonl

ZIP = "http://datasets.vqa.mmsp-kn.de/archives/koniq10k_512x384.zip"

# KonIQ's raters judged one thing (technical image quality on a 1-5 scale), so every phrasing must mean
# exactly that. Varied wording stops the model keying on one fixed string (dataset v2, fix #16); the
# 5-vote human histogram is unchanged. Picked deterministically per image.
PHRASINGS = [
    "How would you rate the quality of this image?",
    "Rate the technical quality of this photo.",
    "How good is the image quality here?",
    "On a scale of 1 to 5, how would you rate this picture's quality?",
    "How would you judge the overall quality of this photograph?",
    "Rate this image's quality, from 1 (bad) to 5 (excellent).",
    "How high is the quality of this picture?",
    "Overall, how good does this photo look in terms of quality?",
    "What quality rating would you give this image?",
    "How would you score the visual quality of this photo?",
    "Judge the image quality of this picture.",
    "How well taken is this photo, quality-wise?",
]
ANCHORS = [["bad", "excellent"], ["very poor", "excellent"], ["poor", "excellent"], ["very low quality", "very high quality"]]


def pick(options, key):
    return options[int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % len(options)]


def main():
    by_image, images = {}, []
    with open(RAW / "koniq/koniq10k_scores_and_distributions.csv") as f:
        for r in csv.DictReader(f):
            image_id = f"koniq_{r['image_name'][:-4]}"
            total = int(r["c_total"])
            by_image[image_id] = [{
                "type": "score", "family": "quality", "source": "koniq", "ref": r["image_name"],
                "q": pick(PHRASINGS, image_id), "scale": [1, 5], "anchors": pick(ANCHORS, image_id + "a"),
                "target": [round(int(r[f"c{i}"]) / total, 4) for i in range(1, 6)], "na": 0.0, "label_kind": "human_soft",
            }]
            images.append({"image_id": image_id, "source": "koniq", "archive": ZIP, "member": r["image_name"],
                           "width": 512, "height": 384, "license": "research use"})
    write_by_image("koniq", by_image)
    write_jsonl(OUT / "images_koniq.jsonl", images)


if __name__ == "__main__":
    main()
