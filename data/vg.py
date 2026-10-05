"""Visual Genome image index -> out/images_vg.jsonl, only for images that are not in COCO (the rest use coco_ ids)."""
from common import OUT, RAW, load, write_jsonl


def main():
    rows = [{"image_id": f"vg_{d['image_id']}", "source": "vg", "url": d["url"],
             "width": d["width"], "height": d["height"], "license": None}
            for d in load(RAW / "vg/image_data.json") if not d["coco_id"]]
    write_jsonl(OUT / "images_vg.jsonl", rows)


if __name__ == "__main__":
    main()
