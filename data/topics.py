"""Topic taxonomy, style exemplars and the deterministic spec planner for teacher-written questions
(dataset v2).

A *spec* fixes what one teacher-written question must be: its kind (bool with an intended Yes/No
answer, choice with an exact option count, score with an exact scale, or a false-premise question
that should come out N/A), a primary topic plus two alternates the writer may switch to when the
primary doesn't fit the image, and the style exemplars shown for phrasing variety. Everything is a
pure function of the base dataset, so re-running gives byte-identical specs.

    uv run python data/topics.py              # plan -> out/v2/specs.jsonl, print the plan's stats
"""
import hashlib
import json
from collections import Counter, defaultdict

from common import OUT, read_jsonl, write_jsonl

# name: (family tag, what to ask about, kinds it supports). Families feed per-family evaluation;
# material/open_closed are the held-out families, so those two topics are planned on test images only.
TOPICS = {
    "presence":       ("presence", "whether a specific object, person or animal is in the scene", "bc"),
    "counting":       ("count", "how many of something there are (small counts)", "bcs"),
    "colour":         ("color", "the colour of a specific object", "bc"),
    "shape":          ("shape", "the shape of a specific object", "bc"),
    "size":           ("size", "how big something is, absolutely or relative to something else", "bcs"),
    "spatial":        ("relation", "where one thing is relative to another (left of, behind, on top of ...)", "bc"),
    "frame_position": ("position", "where in the picture something appears (top, bottom, centre, left, right ...)", "bc"),
    "action":         ("action", "what a person or animal is doing", "bc"),
    "pose":           ("pose", "body posture or gesture (sitting, standing, arms raised ...)", "bc"),
    "clothing":       ("clothing", "what someone is wearing", "bc"),
    "expression":     ("emotion", "facial expression or apparent mood", "bcs"),
    "age":            ("age", "the apparent age group of a person", "bcs"),
    "weather":        ("weather", "the weather conditions", "bcs"),
    "time_of_day":    ("time_of_day", "the time of day", "bc"),
    "season":         ("season", "the season of the year", "bc"),
    "setting":        ("scene", "whether the scene is indoors or outdoors, urban or rural", "bc"),
    "lighting":       ("lighting", "lighting: brightness, light source, shadows", "bcs"),
    "condition":      ("condition", "the state of repair, wear or damage of an object", "bcs"),
    "cleanliness":    ("cleanliness", "how clean, tidy or cluttered something is", "bcs"),
    "text_signage":   ("text", "whether there is text, a sign or a label and what kind (not long readings)", "bc"),
    "safety":         ("safety", "hazards, risks or safety equipment", "bcs"),
    "place":          ("scene", "what kind of place this is", "bc"),
    "food":           ("food", "food and drink: what it is, how it is prepared or served", "bcs"),
    "vehicle":        ("vehicle", "vehicles: type, what they are doing", "bc"),
    "animal":         ("animal", "animals: species, behaviour", "bc"),
    "image_quality":  ("quality", "photographic quality: blur, focus, exposure, noise", "bcs"),
    "composition":    ("composition", "how the photo is taken: viewpoint, angle, close-up or wide", "bc"),
    "crowd":          ("crowd", "how crowded or busy the scene is", "bcs"),
    "purpose":        ("purpose", "what an object is for, or what is likely to happen next", "bc"),
    "motion":         ("motion", "whether things are moving, and how fast", "bcs"),
    "distance":       ("distance", "how far away something is from the camera or from something else", "bcs"),
    "texture":        ("texture", "surface texture: smooth, rough, shiny, patterned (not what it is made of)", "bc"),
    # held-out families: test images only
    "material":       ("material", "what a specific object is made of", "bc"),
    "open_closed":    ("open_closed", "whether something (door, window, laptop, mouth, umbrella ...) is open or closed", "bc"),
}
HELD_OUT_TOPICS = {"material", "open_closed"}
TRAIN_TOPICS = sorted(t for t in TOPICS if t not in HELD_OUT_TOPICS)

# Kind quotas for teacher questions, tuned so train lands at bool 35-40 / choice 30-35 / score 15-20%
# (distinct) once added to the human base (bool 46 / choice 27 / score 27%), with ~10% false-premise
# questions for N/A. na_* count toward their type's share.
# N/A kinds doubled after pilot 3 (2026-10-01): only ~55% of false-premise specs come out N/A-dominant, so
# 0.05 + 0.05 gave 6.6% N/A-dominant train questions against the 8-12% target; 0.095 + 0.095 projects to ~10%.
KIND_WEIGHTS = {"bool_yes": 0.12, "bool_no": 0.12, "choice": 0.32, "score": 0.15, "na_bool": 0.095, "na_choice": 0.095}
OPTION_WEIGHTS = {2: 0.15, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.12, 7: 0.09, 8: 0.09}
SCALES = {(1, 3): 0.2, (1, 5): 0.2, (1, 7): 0.2, (1, 10): 0.2, (0, 10): 0.2}

# Style exemplars (other images, phrasing variety only -- the writer is told not to copy them).
EXEMPLARS = {
    "bool": ["Is anyone holding an umbrella?", "Could this dog be asleep?", "Are both lamps switched on?",
             "Does the man have a beard?", "Would you need a coat to stand here?", "Has it been raining recently?",
             "Are there more cars than bicycles?", "Is the bus pulling away from the kerb?", "Do the curtains look new?",
             "Is the sun behind the photographer?", "Can you see any reflections on the floor?",
             "Was this shot from above?", "Are the two children wearing matching shirts?", "Is the cake partly eaten?",
             "Does the street look busy?", "Is a vehicle parked on the grass?"],
    "choice": ["Which animal is closest to the camera?", "How is the woman carrying her bag?",
               "Where is the clock relative to the window?", "Which season does this look like?",
               "What is the boy most likely about to do?", "How are the plates arranged?",
               "From which direction is the light coming?", "What best describes the sky?",
               "Who seems to be leading the group?", "Which part of the frame is in sharpest focus?",
               "What kind of shop is this?", "How would you describe the man's posture?",
               "In what way is the road surface damaged?", "Which object is the cat looking at?"],
    "score": ["How busy does the platform look?", "How worn is the sofa?", "Rate how sharp this photo is.",
              "How far away is the boat from the shore?", "How tidy is the desk?", "How strong does the wind seem?",
              "How bright is the room?", "How happy does the child look?", "How appetising does the meal look?",
              "How risky does this activity look?", "How old does the building appear?", "How fast is the cyclist going?"],
}


def h(*key):
    """Deterministic 64-bit hash of a key (tie-breaks and exemplar picks; never Python's salted hash())."""
    return int(hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:16], 16)


class Quota:
    """Greedy quota filler: always pick the allowed value furthest below its target share, ties broken
    by a hash of the key. Over many draws the realised mix matches the targets almost exactly, which a
    random sampler wouldn't guarantee."""

    def __init__(self, weights):
        self.w, self.n = dict(weights), Counter()

    def pick(self, key, allowed=None):
        allowed = list(allowed or self.w)
        total = sum(self.n[v] for v in self.w) + 1
        v = max(allowed, key=lambda v: (self.w[v] * total - self.n[v], -h(key, v)))
        self.n[v] += 1
        return v


def kind_type(kind):
    return "bool" if kind in ("bool_yes", "bool_no", "na_bool") else "score" if kind == "score" else "choice"


def n_new(split, existing):
    """New teacher questions for an image with `existing` distinct human questions."""
    return max(1, min(4, 5 - existing)) if split == "train" else 1


def plan(records):
    kinds, options, scales = Quota(KIND_WEIGHTS), Quota(OPTION_WEIGHTS), Quota(SCALES)
    topic_q = Quota({t: 1 / len(TRAIN_TOPICS) for t in TRAIN_TOPICS})
    held_q = Quota({t: 0.5 for t in HELD_OUT_TOPICS})
    out = []
    for r in sorted(records, key=lambda r: r["image_id"]):
        iid, split = r["image_id"], r["split"]
        qs = [q for q in r["questions"] if not q.get("aug")]
        have = {q["type"] for q in qs}
        specs, used = [], set()
        for slot in range(n_new(split, len(qs))):
            key = (iid, slot)
            missing = [k for k in KIND_WEIGHTS if kind_type(k) not in have and not k.startswith("na")]
            kind = kinds.pick(key, missing or None)
            have.add(kind_type(kind))
            spec = {"slot": slot, "kind": kind, "type": kind_type(kind)}
            if spec["type"] == "choice":
                spec["k"] = options.pick(key)
            if kind == "score":
                spec["scale"] = list(scales.pick(key))
            cands = [t for t in TRAIN_TOPICS if t not in used and kind_type(kind)[0] in TOPICS[t][2]]
            primary = topic_q.pick(key, cands)
            alts = sorted((t for t in cands if t != primary), key=lambda t: h(key, "alt", t))[:2]
            used.add(primary)
            spec["topics"] = [primary] + alts
            spec["exemplars"] = sorted(EXEMPLARS[spec["type"]], key=lambda e: h(key, "ex", e))[:3]
            specs.append(spec)
        # held-out families: grow the test sets (open_closed was 198, needs >= 2,500), test images only.
        # Over-planned because many images have nothing open/closable; the writer may skip.
        if split == "test" and h(iid, "held") % 100 < 55:
            topic = held_q.pick((iid, "held"))
            kind = "bool_yes" if h(iid, "pol") % 2 else "bool_no"
            if h(iid, "ht") % 2:
                kind = "choice"
            spec = {"slot": len(specs), "kind": kind, "type": kind_type(kind), "topics": [topic], "held_out": True,
                    "exemplars": sorted(EXEMPLARS[kind_type(kind)], key=lambda e: h(iid, "hex", e))[:3]}
            if kind == "choice":
                spec["k"] = 2 if topic == "open_closed" else options.pick((iid, "held"))
            specs.append(spec)
        out.append({"image_id": iid, "split": split, "existing": [q["q"] for q in qs], "specs": specs})
    return out


def main():
    records = list(read_jsonl(OUT / "v2/base_selected.jsonl"))
    specs = plan(records)
    write_jsonl(OUT / "v2/specs.jsonl", specs)
    all_specs = [(r["split"], s) for r in specs for s in r["specs"]]
    print(f"{len(all_specs):,} specs on {len(specs):,} images")
    for split in ("train", "val", "test"):
        ss = [s for sp, s in all_specs if sp == split]
        print(f"  {split}: {len(ss):,}")
    ss = [s for _, s in all_specs]
    print("kinds:", {k: f"{v / len(ss):.1%}" for k, v in Counter(s["kind"] for s in ss).most_common()})
    ch = [s["k"] for s in ss if "k" in s]
    print("option counts:", {k: f"{v / len(ch):.1%}" for k, v in sorted(Counter(ch).items())})
    sc = [tuple(s["scale"]) for s in ss if "scale" in s]
    print("scales:", {str(k): f"{v / len(sc):.1%}" for k, v in sorted(Counter(sc).items())})
    tp = Counter(s["topics"][0] for s in ss)
    print(f"primary topics: {len(tp)}, max share {max(tp.values()) / len(ss):.1%}, min share {min(tp.values()) / len(ss):.1%}")
    print("held-out specs (test):", Counter(s["topics"][0] for s in ss if s.get("held_out")))
    # expected train mix once added to the human base, before writer/label losses
    base = Counter(q["type"] for r in records if r["split"] == "train" for q in r["questions"] if not q.get("aug"))
    new = Counter(s["type"] for sp, s in all_specs if sp == "train")
    tot = sum(base.values()) + sum(new.values())
    print("expected train mix:", {t: f"{(base[t] + new[t]) / tot:.1%}" for t in ("bool", "choice", "score")},
          f"distinct/image {tot / sum(r['split'] == 'train' for r in records):.2f}")


if __name__ == "__main__":
    main()
