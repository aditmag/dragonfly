"""Paths and small helpers shared by the dataset builders."""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
OUT = ROOT / "out"


def load(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n:,} rows to {path}")


def write_by_image(name, by_image):
    write_jsonl(OUT / f"{name}.jsonl", ({"image_id": k, "questions": v} for k, v in by_image.items()))


def coco_id(i):
    return f"coco_{int(i):012d}"


@lru_cache
def _vg_to_coco():
    return {d["image_id"]: d["coco_id"] for d in load(RAW / "vg/image_data.json")}


def vg_image(vg_id):
    """Our image id for a Visual Genome id: coco_<id> if VG says it is a COCO image, else vg_<id>.
    None if VG doesn't know the id (e.g. GQA's few non-VG images)."""
    m = _vg_to_coco()
    if not str(vg_id).isdigit() or int(vg_id) not in m:
        return None
    return coco_id(m[int(vg_id)]) if m[int(vg_id)] else f"vg_{int(vg_id)}"


def split_of(image_id):
    """train 90% / val 5% / test 5%, by image (POPE images are forced to test in merge.py)."""
    h = int(hashlib.sha1(image_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 90 else "val" if h < 95 else "test"


def be(name):
    """'is' or 'are' for a (crudely detected) plural noun."""
    return "are" if name.endswith("s") and not name.endswith(("ss", "us", "is")) else "is"


# Keyword rules for tagging free-text questions (VQAv2, GQA, ...). First match wins,
# so held-out families come first: a false positive only holds out an extra question.
FAMILY_RULES = [
    ("open_closed", r"\b(open|opened|closed|shut)\b"),
    ("material", r"\b(made of|made from|material|wooden|wood|metal|metallic|plastic|glass|leather|ceramic|"
                 r"steel|concrete|brick|stone|cotton|wool|paper|cardboard)\b"),
    ("count", r"\b(how many|two|three|four|five|six|seven|eight|nine|ten|\d+)\b"),
    ("color", r"\b(colou?rs?|red|blue|green|yellow|white|black|brown|orange|pink|purple|gr[ae]y|silver|gold)\b"),
    ("presence", r"^(is|are) there\b"),
]


def family(question):
    q = question.lower()
    for name, pattern in FAMILY_RULES:
        if re.search(pattern, q):
            return name
    return "other"
