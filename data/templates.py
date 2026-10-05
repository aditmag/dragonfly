"""Template questions from COCO instances and Visual Genome -> out/templates.jsonl.

  tpl_presence  bool    "Is there any dog here?" on COCO images. Half yes; half no, absent category drawn by
                        co-occurrence (hard negatives, e.g. a fork missing from a kitchen) or uniformly.
  tpl_na        bool/choice  presupposes an absent COCO object ("What colour is the car?" with no car) -> na = 1.
                        Only COCO's 80 categories count as reliable absence; VG labels aren't exhaustive.
  tpl_relation  bool    "Is the man riding the horse?" from VG relationships. Half yes; the no is the same
                        asymmetric relation with subject and object swapped.
  tpl_colour    choice  "What colour is the shirt?" from VG attributes (objects with exactly one colour annotated).
  tpl_material  choice  same for materials; generated for test-split images only, since it is a held-out family.

Objects are only asked about when their name is unique in the image ("the shirt" must be unambiguous).
VG attribute yes/no is skipped on purpose: an attribute missing from a VG object doesn't mean it's false.
"""
import random
import re
from collections import Counter, defaultdict

from common import RAW, be, coco_id, load, split_of, vg_image, write_by_image

N_PRESENCE = 15_000     # images, each with one yes and one no question
N_NA = 12_500           # images, each with one choice and one bool N/A question
N_RELATION = 15_000
N_COLOUR = 24_000

COLOURS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray"]
MATERIALS = {"wooden": "wood", "wood": "wood", "metal": "metal", "metallic": "metal", "plastic": "plastic",
             "glass": "glass", "brick": "brick", "leather": "leather", "concrete": "concrete", "cement": "concrete",
             "stone": "stone", "paper": "paper", "ceramic": "ceramic", "cloth": "fabric", "fabric": "fabric"}
# Asymmetric relations that read naturally as "Is the {s} {p} the {o}?" and are false when swapped.
RELATIONS = {"on", "on top of", "in front of", "behind", "above", "under", "wearing", "holding", "riding",
             "carrying", "sitting on", "standing on", "eating", "looking at", "hanging from", "parked on"}
PRESENCE = ["Is there any {c} here?", "Can you see any {c}?", "Does this picture have any {c}?", "Is any {c} visible?"]

rng = random.Random(0)


def question(type_, family, source, q, target, na=0.0, **extra):
    return {"type": type_, "family": family, "source": source, "q": q, "target": target, "na": na,
            "label_kind": "exact", **extra}


def bool_(family, source, q, yes):
    return question("bool", family, source, q, {"yes": float(yes), "no": float(not yes)})


def choice(family, source, q, options, answer):
    return question("choice", family, source, q, {o: float(o == answer) for o in options}, options=options)


def coco_questions(by_image):
    cats = {}
    image_cats = defaultdict(set)
    for split in ("train2017", "val2017"):
        d = load(RAW / f"coco/annotations/instances_{split}.json")
        cats.update({c["id"]: c["name"] for c in d["categories"]})
        for a in d["annotations"]:
            image_cats[a["image_id"]].add(cats[a["category_id"]])
    names = sorted(cats.values())
    cooc = defaultdict(Counter)
    for present in image_cats.values():
        for a in present:
            cooc[a].update(present)

    def absent(present, hard):
        others = [c for c in names if c not in present]
        return rng.choices(others, [1 + sum(cooc[p][c] for p in present) for c in others])[0] if hard else rng.choice(others)

    images = sorted(image_cats)
    for i in rng.sample(images, N_PRESENCE):
        present = sorted(image_cats[i])
        out = by_image[coco_id(i)]
        out.append(bool_("presence", "tpl_presence", rng.choice(PRESENCE).format(c=rng.choice(present)), True))
        out.append(bool_("presence", "tpl_presence", rng.choice(PRESENCE).format(c=absent(present, rng.random() < 0.5)), False))
    for i in rng.sample(images, N_NA):
        present = sorted(image_cats[i])
        out = by_image[coco_id(i)]
        c = absent(present, False)
        options = rng.sample(COLOURS, 4)
        out.append(question("choice", "absent_object", "tpl_na", f"What colour {be(c)} the {c}?", None, na=1.0, options=options))
        c = absent(present, False)
        out.append(question("bool", "absent_object", "tpl_na", f"{be(c).capitalize()} the {c} {rng.choice(COLOURS)}?", None, na=1.0))


def vg_questions(by_image):
    name_counts = {d["image_id"]: Counter(o["names"][0].lower().strip() for o in d["objects"] if o["names"])
                   for d in load(RAW / "vg/objects.json")}

    relations = []
    for d in load(RAW / "vg/relationships.json"):
        counts, pairs = name_counts[d["image_id"]], []
        for r in d["relationships"]:
            s = (r["subject"].get("name") or r["subject"]["names"][0]).lower().strip()
            o = (r["object"].get("name") or r["object"]["names"][0]).lower().strip()
            if r["predicate"].lower().strip() in RELATIONS and s != o and counts[s] == 1 and counts[o] == 1:
                pairs.append((s, r["predicate"].lower().strip(), o))
        if pairs:
            relations.append((d["image_id"], rng.choice(pairs)))
    for image, (s, p, o) in rng.sample(relations, min(N_RELATION, len(relations))):
        yes = rng.random() < 0.5
        s, o = (s, o) if yes else (o, s)
        by_image[vg_image(image)].append(bool_("relation", "tpl_relation", f"{be(s).capitalize()} the {s} {p} the {o}?", yes))

    colours, materials = [], []
    for d in load(RAW / "vg/attributes.json"):
        counts = name_counts[d["image_id"]]
        found = {"colour": [], "material": []}
        for o in d["attributes"]:
            if not o.get("names") or counts[o["names"][0].lower().strip()] != 1:
                continue
            name = o["names"][0].lower().strip()
            attrs = {a.lower().strip().replace("grey", "gray") for a in o.get("attributes", [])}
            c, m = attrs & set(COLOURS), {MATERIALS[a] for a in attrs if a in MATERIALS}
            if len(c) == 1:
                found["colour"].append((name, next(iter(c))))
            if len(m) == 1:
                found["material"].append((name, next(iter(m))))
        if found["colour"]:
            colours.append((d["image_id"], rng.choice(found["colour"])))
        if found["material"] and split_of(vg_image(d["image_id"])) == "test":
            materials.append((d["image_id"], rng.choice(found["material"])))
    # VG object names can carry the answer ("blue table", "wooden chair"): skip those questions, but only
    # after the random draws, so every other template question stays identical to v1 (dataset v2, fix #14).
    def names_answer(name, vocab):
        return bool(set(re.findall(r"[a-z]+", name.replace("grey", "gray"))) & vocab)  # "orange/black socks"

    colour_words, material_words = set(COLOURS) | {"silver", "gold", "tan", "beige"}, set(MATERIALS)
    for image, (name, colour) in rng.sample(colours, min(N_COLOUR, len(colours))):
        options = [colour] + rng.sample([c for c in COLOURS if c != colour], 3)
        rng.shuffle(options)
        if not names_answer(name, colour_words):
            by_image[vg_image(image)].append(choice("color", "tpl_colour", f"What colour {be(name)} the {name}?", options, colour))
    for image, (name, material) in materials:
        options = [material] + rng.sample([m for m in sorted(set(MATERIALS.values())) if m != material], 3)
        rng.shuffle(options)
        if not names_answer(name, material_words):
            by_image[vg_image(image)].append(choice("material", "tpl_material", f"What {be(name)} the {name} made of?", options, material))


def main():
    by_image = defaultdict(list)
    coco_questions(by_image)
    vg_questions(by_image)
    print(Counter(q["source"] for qs in by_image.values() for q in qs))
    write_by_image("templates", by_image)


if __name__ == "__main__":
    main()
