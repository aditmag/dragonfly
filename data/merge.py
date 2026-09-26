"""Join builder outputs by image -> out/dataset.jsonl + out/images.jsonl, and print stats.

Applies the dataset rules from PLAN.md §7: split by image hash, POPE images only in test,
held-out families only in test, one copy of each question per image.
"""
import hashlib
import math
from collections import Counter, defaultdict

from common import OUT, family, read_jsonl, write_jsonl

SOURCES = ["vqav2", "pope"]      # builder outputs, out/<name>.jsonl
IMAGE_INDEXES = ["coco"]         # out/images_<name>.jsonl
HELD_OUT = {"open_closed", "material"}
LABEL_KINDS = {"exact", "human_soft", "teacher_soft"}


def split_of(image_id):
    h = int(hashlib.sha1(image_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 90 else "val" if h < 95 else "test"


def close_to_one(x):
    return math.isclose(x, 1.0, abs_tol=1e-3)


def problem(q):
    """Why a question is invalid, or None."""
    if not isinstance(q.get("q"), str) or not q["q"].strip():
        return "empty question"
    if q.get("label_kind") not in LABEL_KINDS:
        return "bad label_kind"
    if not 0.0 <= q.get("na", -1) <= 1.0:
        return "bad na"
    t = q.get("target")
    if q["type"] == "bool":
        ok = isinstance(t, dict) and set(t) == {"yes", "no"}
    elif q["type"] == "choice":
        opts = q.get("options", [])
        ok = len(opts) >= 2 and len(set(opts)) == len(opts) and isinstance(t, dict) and set(t) == set(opts)
    elif q["type"] == "score":
        lo, hi = q.get("scale", (0, -1))
        ok = isinstance(t, list) and len(t) == hi - lo + 1
    else:
        return f"unknown type {q['type']!r}"
    if not ok:
        return f"bad {q['type']} target"
    values = t.values() if isinstance(t, dict) else t
    if min(values) < 0 or not close_to_one(sum(values)):
        return "target is not a distribution"
    return None


def main():
    manifest = {}
    for name in IMAGE_INDEXES:
        for row in read_jsonl(OUT / f"images_{name}.jsonl"):
            manifest[row["image_id"]] = row

    by_image = defaultdict(list)
    for name in SOURCES:
        for rec in read_jsonl(OUT / f"{name}.jsonl"):
            by_image[rec["image_id"]].extend(rec["questions"])
    pope_images = {i for i, qs in by_image.items() if any(q["source"].startswith("pope") for q in qs)}

    dropped = Counter()
    missing_images = 0
    records, images = [], []
    for image_id, qs in by_image.items():
        if image_id not in manifest:
            missing_images += 1
            dropped["image not in any index"] += len(qs)
            continue
        split = "test" if image_id in pope_images else split_of(image_id)
        seen, kept = set(), []
        for q in qs:
            reason = problem(q)
            # POPE's three splits repeat questions on purpose; keep each split whole.
            key = (q["q"].strip().lower(), q["type"], q["source"] if q["source"].startswith("pope") else "")
            if reason:
                dropped[reason] += 1
            elif key in seen:
                dropped["duplicate on image"] += 1
            elif split != "test" and HELD_OUT & {q["family"], family(q["q"])}:
                dropped["held-out family outside test"] += 1
            else:
                seen.add(key)
                kept.append(q)
        if kept:
            records.append({"image_id": image_id, "split": split, "questions": kept})
            images.append({**manifest[image_id], "split": split})

    write_jsonl(OUT / "dataset.jsonl", records)
    write_jsonl(OUT / "images.jsonl", images)
    print_stats(records, dropped, missing_images)


def print_stats(records, dropped, missing_images):
    by_split, by_type, by_source, by_family, by_kind = Counter(), Counter(), Counter(), Counter(), Counter()
    images = Counter(r["split"] for r in records)
    per_image = Counter()
    for r in records:
        per_image[min(len(r["questions"]), 10)] += 1
        for q in r["questions"]:
            by_split[r["split"]] += 1
            by_type[(r["split"], q["type"])] += 1
            by_source[q["source"]] += 1
            by_family[(q["family"], r["split"])] += 1
            by_kind[q["label_kind"]] += 1
    total = sum(by_split.values())

    print(f"\n{total:,} questions on {len(records):,} images")
    print("\nsplit      images   questions")
    for s in ("train", "val", "test"):
        print(f"  {s:<7} {images[s]:>8,} {by_split[s]:>11,}")
    print("\nby type:", ", ".join(f"{s}/{t}: {n:,}" for (s, t), n in sorted(by_type.items())))
    print("by source:", ", ".join(f"{k}: {n:,}" for k, n in by_source.most_common()))
    print("by label kind:", ", ".join(f"{k}: {n:,}" for k, n in by_kind.most_common()))
    print("\nfamily          train      val     test")
    for fam in sorted({f for f, _ in by_family}):
        print(f"  {fam:<12} {by_family[(fam, 'train')]:>7,} {by_family[(fam, 'val')]:>8,} {by_family[(fam, 'test')]:>8,}")
    print("\nquestions per image:", ", ".join(f"{k}{'+' if k == 10 else ''}: {n:,}" for k, n in sorted(per_image.items())))
    print("\ndropped:", ", ".join(f"{k}: {n:,}" for k, n in dropped.most_common()) or "none")
    if missing_images:
        print(f"images missing from every index: {missing_images:,}")


if __name__ == "__main__":
    main()
