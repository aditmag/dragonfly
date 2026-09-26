"""Paths and small helpers shared by the dataset builders."""
import json
import re
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
    print(f"wrote {n:,} rows to {path.relative_to(ROOT.parent)}")


def coco_id(i):
    return f"coco_{int(i):012d}"


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
