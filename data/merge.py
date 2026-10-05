"""Join builder outputs by image -> out/dataset.jsonl + out/images.jsonl, and print stats.

Applies the dataset rules: split by image hash, POPE images only in test,
held-out families only in test, one copy of each question per image.
"""
import math
import re
from collections import Counter, defaultdict

from common import OUT, family, read_jsonl, split_of, write_jsonl

# builder outputs, out/<name>.jsonl
SOURCES = ["vqav2", "vqav2_choice", "gqa", "aokvqa", "tallyqa", "koniq", "vizwiz", "templates", "pope",
           "vlm_v2"]  # teacher-written (teacher.py finalize)
OPTIONAL = {"vlm_v2"}  # skipped if not built yet (the human-only base the teacher run is planned from)
IMAGE_INDEXES = ["coco", "vg", "vizwiz", "koniq"]   # out/images_<name>.jsonl
HELD_OUT = {"open_closed", "material"}
LABEL_KINDS = {"exact", "human_soft", "teacher_soft"}


def close_to_one(x):
    return math.isclose(x, 1.0, abs_tol=1e-3)


EITHER_OR = re.compile(r"\b\w+ or \w+\b")  # bool questions matching this need a yes/no classifier verdict
# teacher.py classify over every human either/or bool in the unfiltered pool (out/v2/human_either_or.jsonl), so
# any selection is covered; teacher-written ones are handled in teacher.py label
VERDICTS = OUT / "v2/human_verdicts.jsonl"

DENIALS = {"unknown", "unclear", "not sure", "not possible", "cannot tell", "can't tell", "cannot determine",
           "can't determine", "none of the above"}


def _stem(w):
    return w[:-2] if w.endswith("es") and len(w) > 4 else w[:-1] if w.endswith("s") and len(w) > 3 else w


def is_denial(option, question):
    """An option that denies the question's premise rather than answering it: "no door" for "What colour is
    the door?", or "unknown". Those belong in na, not among the answers (dataset v2, fix #15). "none",
    "nothing", "no one" and sign texts like "no parking" stay: they answer the question."""
    o = option.strip().lower()
    if o in DENIALS:
        return True
    m = re.fullmatch(r"no ([a-z]+)", o)
    return bool(m) and _stem(m.group(1)) in {_stem(w) for w in re.findall(r"[a-z]+", question.lower())}


def move_denials_to_na(q):
    """Remove premise-denial options from a choice question, moving their target mass into na. Returns
    the fixed question, or None if fewer than 2 options remain."""
    if q["type"] != "choice":
        return q
    bad = [o for o in q["options"] if is_denial(o, q["q"])]
    if not bad:
        return q
    keep = [o for o in q["options"] if o not in bad]
    if len(keep) < 2:
        return None
    t, na = q["target"], q["na"]
    if t is not None:
        moved = sum(t[o] for o in bad)
        na = na + (1 - na) * moved
        rest = sum(t[o] for o in keep)
        t = {o: t[o] / rest for o in keep} if rest > 0 and na < 1 else None
        na = 1.0 if t is None else na
    return {**q, "options": keep, "target": t, "na": na}


def problem(q):
    """Why a question is invalid, or None."""
    if not isinstance(q.get("q"), str) or not q["q"].strip():
        return "empty question"
    if q.get("label_kind") not in LABEL_KINDS:
        return "bad label_kind"
    if not 0.0 <= q.get("na", -1) <= 1.0:
        return "bad na"
    t = q.get("target")
    if (t is None) != (q["na"] == 1.0):   # not applicable: no target, the loss is BCE on the N/A output only
        return "target must be null exactly when na = 1"
    if q["type"] == "bool":
        ok = t is None or (isinstance(t, dict) and set(t) == {"yes", "no"})
    elif q["type"] == "choice":
        opts = q.get("options", [])
        ok = len(opts) >= 2 and len(set(opts)) == len(opts) and (t is None or (isinstance(t, dict) and set(t) == set(opts)))
    elif q["type"] == "score":
        lo, hi = q.get("scale", (0, -1))
        ok = hi > lo and (t is None or (isinstance(t, list) and len(t) == hi - lo + 1))
    else:
        return f"unknown type {q['type']!r}"
    if not ok:
        return f"bad {q['type']} target"
    if t is None:
        return None
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
    denials = Counter()
    # human bool questions phrased "X or Y": keep only those the classifier says are plain yes/no
    # (teacher questions were already converted or verified in teacher.py label)
    verdicts = {(v["image_id"], v["q"]): v["yesno_ok"] for v in read_jsonl(VERDICTS)} if VERDICTS.exists() else {}
    for name in SOURCES:
        if name in OPTIONAL and not (OUT / f"{name}.jsonl").exists():
            print(f"{name}: not built yet, skipped")
            continue
        for rec in read_jsonl(OUT / f"{name}.jsonl"):
            for q in rec["questions"]:
                if q["type"] == "bool" and EITHER_OR.search(q["q"]) and q["label_kind"] != "teacher_soft":
                    ok = verdicts.get((rec["image_id"], q["q"]))
                    if ok is False:
                        denials["either/or bool dropped"] += 1
                        continue
                    if ok:
                        q = {**q, "yesno_ok": True}
                fixed = move_denials_to_na(q)
                if fixed is not q:
                    denials["dropped (<2 options left)" if fixed is None else "denial option moved to na"] += 1
                if fixed is not None:
                    by_image[rec["image_id"]].append(fixed)
    print("premise-denial / either-or fixes:", dict(denials) or "none")
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
