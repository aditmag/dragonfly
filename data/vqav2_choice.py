"""VQAv2 open-ended ('other') questions -> multiple choice with human-spread labels -> out/vqav2_choice.jsonl.

Only question types with a compact answer vocabulary (colours, rooms, sports, ...) are converted: for the diffuse
ones ("what is the ...") random distractors are nonsense and the question is trivially easy.

Options: the answers at least 2 humans gave (max 3, most frequent first, dropping ones that overlap the top answer,
e.g. "ground" / "on ground") plus distractors from the common answers of the same question type that share no word
with any human answer, shuffled. Target = human answer counts over the real options; distractors get 0.
"""
import random
from collections import Counter, defaultdict

from common import RAW, coco_id, family, load, write_by_image

MIN_MODE = 4      # skip questions without a clear majority answer
N_OPTIONS = 4
POOL = 50         # distractors come from the 50 commonest answers of the question type
MIN_TYPE = 500    # skip question types with fewer questions...
MIN_COVER = 0.6   # ...or whose top-POOL answers cover less than this share of them


def words(answer):
    return set(answer.split())


def main():
    rows = []
    for split in ("train2014", "val2014"):
        questions = {q["question_id"]: q["question"]
                     for q in load(RAW / f"vqav2/v2_OpenEnded_mscoco_{split}_questions.json")["questions"]}
        for a in load(RAW / f"vqav2/v2_mscoco_{split}_annotations.json")["annotations"]:
            if a["answer_type"] == "other":
                rows.append((a, questions[a["question_id"]]))

    pool = defaultdict(Counter)
    for a, _ in rows:
        pool[a["question_type"]][a["multiple_choice_answer"]] += 1
    top = {t: dict(c.most_common(POOL)) for t, c in pool.items()
           if sum(c.values()) >= MIN_TYPE and sum(n for _, n in c.most_common(POOL)) >= MIN_COVER * sum(c.values())}
    print(f"{len(top)} of {len(pool)} question types kept: {sorted(top)}")

    by_image, skipped = defaultdict(list), 0
    for a, q in rows:
        counts = Counter(x["answer"].strip().lower() for x in a["answers"])
        mode, mode_n = counts.most_common(1)[0]
        if a["question_type"] not in top or mode_n < MIN_MODE or mode not in top[a["question_type"]] or " or " in q.lower():
            skipped += 1
            continue
        real = [mode] + [ans for ans, n in counts.most_common() if n >= 2 and not words(ans) & words(mode)][:2]
        human_words = set().union(*(words(x) for x in counts))
        common = [(ans, n) for ans, n in top[a["question_type"]].items() if not words(ans) & human_words]
        if len(common) < N_OPTIONS - len(real):
            skipped += 1
            continue
        rng = random.Random(a["question_id"])
        distractors = []
        while len(distractors) < N_OPTIONS - len(real):
            pick = rng.choices([c[0] for c in common], [c[1] for c in common])[0]
            if pick not in distractors:
                distractors.append(pick)
        options = real + distractors
        rng.shuffle(options)
        total = sum(counts[o] for o in real)
        by_image[coco_id(a["image_id"])].append({
            "type": "choice", "family": family(q), "source": "vqav2_choice", "ref": a["question_id"],
            "q": q, "options": options, "target": {o: round(counts[o] / total, 4) if o in real else 0.0 for o in options},
            "na": 0.0, "label_kind": "human_soft",
        })
    print(f"vqav2 choice: {sum(map(len, by_image.values())):,} questions, {skipped:,} skipped")
    write_by_image("vqav2_choice", by_image)


if __name__ == "__main__":
    main()
