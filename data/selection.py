"""Select + rebalance out/dataset.jsonl for training -> out/dataset_selected.jsonl (+ images_selected.jsonl).

Keeps all of val/test (already small; they're the eval set, not a diversity budget). Caps
train per source so no single high-volume source (tallyqa's templates, gqa, vqav2) crowds
out the rest. For choice questions kept in train, adds one lower-option-count
variant: winner + one loser, so the model sees option counts other than 4.

# ponytail: only drops to k=2, no k=3 or upward (5-8) variants -- those need a distractor
# pool the builders don't record. Add k=3 the same way if k=2 alone proves too easy; add
# 5-8 by having vqav2_choice/aokvqa/templates stash a few unused distractor candidates per
# question for this script to draw on.
"""
import json
import random
from collections import Counter, defaultdict

from common import OUT, read_jsonl, write_jsonl
from topics import h

# train-only caps by source prefix (tallyqa is split into tallyqa_amt/imported_genome/imported_vqa/
# generate/tdiuc_templates -- capped as one group so no single data_source eats the whole budget).
# A source below its cap is kept whole. Tuned so the total lands at ~250-300k without starving the small, high-value sources (aokvqa, tpl_relation, koniq,
# vizwiz all keep their full train share).
TRAIN_CAPS = {
    "vqav2": 40_000,
    "tallyqa": 60_000,
    "gqa": 40_000,
    "vqav2_choice": 30_000,
    "tpl_presence": 15_000,
    "tpl_na": 15_000,
    "tpl_colour": 15_000,
}

SEED = 0


def cap_group(source):
    """The TRAIN_CAPS key a source counts against (tallyqa_* all share one cap)."""
    return "tallyqa" if source.startswith("tallyqa") else source


AUG_SHARE = 0.5  # dataset v2 (fix #17): half of eligible questions get one variant, was all of them


def augmented(q, rng):
    """A smaller-option variant (winner + 1 or 2 losers) of a human/exact-labelled choice question, or None.
    Teacher-labelled (VLM) questions are excluded: dataset v2 writes them with spec'd option counts 2-8
    already, so copies would only add duplicate content. One rng draw per eligible question either way."""
    if q["type"] != "choice" or q["target"] is None or len(q["options"]) <= 2 or q["label_kind"] == "teacher_soft":
        return None
    if rng.random() >= AUG_SHARE:
        return None
    k = rng.choice([2, 3]) if len(q["options"]) > 3 else 2
    winner = max(q["target"], key=q["target"].get)
    losers = rng.sample([o for o in q["options"] if o != winner], k - 1)
    # Renormalise over the kept options rather than forcing 1.0/0.0: vqav2_choice's human_soft targets
    # can give a real second answer partial credit, and that ratio should survive the drop (a true
    # distractor already has target 0, so this reduces to 1.0/0.0 there).
    options = [winner] + losers
    total = sum(q["target"][o] for o in options)
    rng.shuffle(options)
    variant = {**q, "options": options, "target": {o: q["target"][o] / total for o in options}}
    if "ref" in q:  # not every builder sets one (e.g. templates.py); leave it out rather than fake it
        variant["ref"] = f"{q['ref']}_k{k}"
    variant["aug"] = True  # so stats can count distinct questions without parsing refs
    return variant


TOPIC_CAP = 0.08     # teacher questions: no topic above 8% of train teacher questions (dataset v2, §F)
YES_RATE_MAX = 0.55  # teacher bool questions: mean P(yes) brought down to 0.50 if above this
# teacher questions: no 4-word opening above 0.8% of train (qa.py target <= 1% over all splits; val/test aren't
# capped, so 0.9% landed at 1.01% overall in the full run)
OPENING_CAP = 0.008


def rng_for(*key):
    """A random stream per decision, so selecting one source never depends on which other sources are
    present: the human selection is identical with or without the teacher data (dataset v2 needs this,
    since the teacher's per-image question counts are planned from the human selection)."""
    return random.Random("|".join(map(str, (SEED,) + key)))


def balance_teacher(selected):
    """Drop train teacher questions (§F): topic over TOPIC_CAP, then yes-majority bools until the mean
    P(yes) is 0.50. Drops prefer images with the most distinct questions, so per-image minimums hold."""
    items = [(i, q) for i, qs in selected.items() for q in qs if q["label_kind"] == "teacher_soft"]
    if not items:
        return Counter()
    density = Counter(i for i, qs in selected.items() for q in qs if not q.get("aug"))
    drop, why = set(), Counter()

    def pick(cands, n, key):  # n candidates, from the densest images first, ties broken by a seeded shuffle
        cands = sorted(cands, key=lambda x: (x[1]["q"], x[0]))
        rng_for("balance", key).shuffle(cands)
        cands.sort(key=lambda x: -density[x[0]])
        return cands[:n]

    by_topic = defaultdict(list)
    for i, q in items:
        by_topic[q.get("topic", q["family"])].append((i, q))
    cap = int(TOPIC_CAP * len(items))
    for topic, its in sorted(by_topic.items()):
        for i, q in pick(its, len(its) - cap, ("topic", topic)) if len(its) > cap else []:
            drop.add(id(q)); density[i] -= 1; why["topic cap"] += 1

    # phrasing: no 4-word opening above OPENING_CAP (pilot 1: "how bright is the" 2.6%, "is the photo taken" 2.5%)
    by_open = defaultdict(list)
    for i, q in items:
        if id(q) not in drop:
            by_open[" ".join(q["q"].lower().split()[:4])].append((i, q))
    cap = int(OPENING_CAP * len(items))
    for opening, its in sorted(by_open.items()):
        for i, q in pick(its, len(its) - cap, ("opening", opening)) if len(its) > cap else []:
            drop.add(id(q)); density[i] -= 1; why["opening cap"] += 1

    bools = [(i, q) for i, q in items if q["type"] == "bool" and q["target"] and id(q) not in drop]
    p_yes = lambda q: q["target"]["yes"] / (q["target"]["yes"] + q["target"]["no"])
    total = sum(p_yes(q) for _, q in bools)
    if bools and total / len(bools) > YES_RATE_MAX:
        yes_heavy = [(i, q) for i, q in bools if p_yes(q) > 0.5]
        n, k = len(bools), 0
        for i, q in pick(yes_heavy, len(yes_heavy), ("yes",)):
            if total / n <= 0.50:
                break
            total -= p_yes(q); n -= 1; k += 1
            drop.add(id(q)); density[i] -= 1
        why["yes-rate balance"] = k

    for i in selected:
        selected[i] = [q for q in selected[i] if id(q) not in drop]
    return why


def main():
    records = list(read_jsonl(OUT / "dataset.jsonl"))

    by_group = defaultdict(list)
    for r in records:
        if r["split"] == "train":
            for q in r["questions"]:
                by_group[cap_group(q["source"])].append((r["image_id"], q))

    dropped = Counter()
    selected = defaultdict(list)
    for group in sorted(by_group):
        items = by_group[group]
        cap = TRAIN_CAPS.get(group)
        if cap is not None and len(items) > cap:
            dropped[group] = len(items) - cap
            # bottom-k by a per-question hash, not rng.sample over the pool: removing n questions from the pool
            # changes at most n picks. rng.sample reshuffled ~4.6k vqav2 images when merge dropped 18 either/or
            # questions (pilot 3), which would have moved images out from under the teacher's spec plan.
            items = sorted(items, key=lambda x: h("cap", group, x[0], x[1]["q"], x[1]["source"]))[:cap]
        for image_id, q in items:
            selected[image_id].append(q)
    balanced = balance_teacher(selected)

    out_records, n_aug = [], 0
    for r in records:
        if r["split"] != "train":
            out_records.append(r)
            continue
        qs = selected.get(r["image_id"])
        if not qs:
            continue
        qs = sorted(qs, key=lambda q: (q["source"], q["q"], json.dumps(q.get("options"))))  # order-independent
        extra = [a for q in qs if (a := augmented(q, rng_for("aug", r["image_id"], q["source"], q["q"])))]
        n_aug += len(extra)
        out_records.append({**r, "questions": qs + extra})

    kept_ids = {r["image_id"] for r in out_records}
    images = [row for row in read_jsonl(OUT / "images.jsonl") if row["image_id"] in kept_ids]
    write_jsonl(OUT / "dataset_selected.jsonl", out_records)
    write_jsonl(OUT / "images_selected.jsonl", images)
    for split in ("train", "val", "test"):  # image lists for model/train.py and model/evaluate.py
        (OUT / f"ids_{split}.txt").write_text("".join(r["image_id"] + "\n" for r in out_records if r["split"] == split))

    total = sum(len(r["questions"]) for r in out_records)
    by_split = Counter(r["split"] for r in out_records)
    print(f"\n{total:,} questions on {len(out_records):,} images (train capped, val/test untouched)")
    print("images by split:", dict(by_split))
    print("dropped by cap (train):", dict(dropped) or "none")
    print("teacher balancing (train):", dict(balanced) or "none")
    print(f"option-count augmentation: {n_aug:,} variants added (train only)")


if __name__ == "__main__":
    main()
