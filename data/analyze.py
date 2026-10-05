"""Diversity report on out/dataset.jsonl: what the data covers, beyond how much of it there is."""
import re
from collections import Counter, defaultdict

from common import OUT, read_jsonl

TOPICS = {
    "weather/time": r"\b(weather|sunny|cloudy|rain|raining|snow|night|daytime|morning|evening|season|winter|summer)\b",
    "lighting": r"\b(bright|dark|lighting|shadow|sunlight)\b",
    "condition/state": r"\b(clean|dirty|old|new|broken|damaged|worn|empty|full|crowded|messy|rusty)\b",
    "emotion/attitude": r"\b(happy|sad|angry|smil\w*|emotion|feel\w*|excited|bored|friendly)\b",
    "size": r"\b(large|small|big|tall|short|long|huge|tiny)\b",
    "image style": r"\b(blurry|sharp|painting|cartoon|drawing|black and white|photograph|photo)\b",
    "text/OCR": r"\b(say|says|written|read|sign|text|word|letter|number)\b",
    "spatial": r"\b(left|right|behind|in front|next to|above|below|under|between)\b",
    "activity": r"\b(playing|riding|eating|holding|wearing|sitting|standing|walking|running|carrying)\b",
}


def main():
    per_source = defaultdict(lambda: {"n": 0, "strings": Counter(), "soft": 0})
    options, scales, na_by = Counter(), Counter(), Counter()
    topics, topics_score = Counter(), Counter()
    by_type, images = Counter(), defaultdict(set)
    n_train = 0
    for rec in read_jsonl(OUT / "dataset.jsonl"):
        for q in rec["questions"]:
            s = per_source[q["source"]]
            s["n"] += 1
            s["strings"][q["q"].lower().strip()] += 1
            t = q["target"]
            if t is not None and max(t.values() if isinstance(t, dict) else t) < 0.999:
                s["soft"] += 1
            images[q["source"]].add(rec["image_id"])
            if q["type"] == "choice":
                options[len(q["options"])] += 1
            if q["type"] == "score":
                scales[(tuple(q["scale"]), tuple(q["anchors"]))] += 1
            if q["na"] == 1.0:
                na_by[(q["source"], rec["split"])] += 1
            if rec["split"] == "train":
                n_train += 1
                by_type[q["type"]] += 1
                text = q["q"].lower()
                for name, pattern in TOPICS.items():
                    if re.search(pattern, text):
                        topics[name] += 1
                        topics_score[name] += q["type"] == "score"

    print("source                    questions  images  distinct-q  top-50-share  soft-label")
    for name, s in sorted(per_source.items(), key=lambda kv: -kv[1]["n"]):
        top50 = sum(n for _, n in s["strings"].most_common(50)) / s["n"]
        print(f"{name:<24} {s['n']:>10,} {len(images[name]):>7,} {len(s['strings']):>11,} {top50:>12.0%} {s['soft'] / s['n']:>11.0%}")
    total_distinct = len(set().union(*(s["strings"].keys() for s in per_source.values())))
    print(f"\ndistinct question strings overall: {total_distinct:,} of {sum(s['n'] for s in per_source.values()):,}")
    print("\nchoice questions by number of options:", dict(sorted(options.items())))
    print("score questions by (scale, anchors):")
    for k, n in scales.most_common():
        print(f"  {k}: {n:,}")
    print("N/A questions by (source, split):", dict(sorted(na_by.items())))
    print(f"\ntopic keywords in train questions (of {n_train:,}):")
    for name in TOPICS:
        print(f"  {name:<18} {topics[name]:>8,} ({topics[name] / n_train:.1%})  of which score questions: {topics_score[name]:,}")


if __name__ == "__main__":
    main()
