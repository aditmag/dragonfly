"""COCO 2017 image index -> out/images_coco.jsonl (URL, size, licence per image).

COCO 2017 train+val holds every 2014 image under the same ids, so this also covers VQAv2 and POPE.
"""
from common import OUT, RAW, coco_id, load, write_jsonl


def main():
    rows = []
    for split in ("train2017", "val2017"):
        d = load(RAW / f"coco/annotations/instances_{split}.json")
        licenses = {l["id"]: l["url"] for l in d["licenses"]}
        for im in d["images"]:
            rows.append({
                "image_id": coco_id(im["id"]),
                "source": "coco",
                "url": im["coco_url"],
                "width": im["width"],
                "height": im["height"],
                "license": licenses[im["license"]],
            })
    write_jsonl(OUT / "images_coco.jsonl", rows)


if __name__ == "__main__":
    main()
