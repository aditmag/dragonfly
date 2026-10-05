"""Dataset acceptance checks (dataset v2). Deterministic, read-only.

    uv run python data/qa.py out/v1/dataset_selected.jsonl                       # the "before" column
    uv run python data/qa.py out/dataset_selected.jsonl --teacher out/teacher_calibration.json --json out/qa_v2.json

Every acceptance target is one check -> PASS / FAIL / MISSING (the data needed to evaluate it doesn't
exist in this file, e.g. v1 has no rotation-stability scores). "Teacher" questions = label_kind
teacher_soft; "distinct" = excluding k-option augmentation copies.
"""
import argparse
import json
import re
from collections import Counter

from merge import EITHER_OR, HELD_OUT, is_denial, problem

OPTION_SPEC = {2: 0.15, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.12, 7: 0.09, 8: 0.09}
# An option that *is* an N/A statement. Anchored: "can't walk" / "can't find" are real answers (A-OKVQA).
NA_OPTION = re.compile(r"^((cannot|can't|unable to|impossible to) (tell|determine|say|know|be determined|be seen)\b.*"
                       r"|not (sure|visible|determinable|clear|applicable|known)"
                       r"|unknown|unclear|undetermined|indeterminate|n/a|none of the above|not enough information)$", re.I)


def is_aug(q):
    return q.get("aug") or re.search(r"_k[23]$", str(q.get("ref", "")))


def argmax(t):
    return max(t, key=t.get) if isinstance(t, dict) else max(range(len(t)), key=t.__getitem__)


def run(path, teacher=None, only=None):
    rows = [json.loads(l) for l in open(path)]
    if only:  # pilot: judge only the images the teacher ran on
        keep = {json.loads(l)["image_id"] for l in open(only)}
        rows = [r for r in rows if r["image_id"] in keep]
    cal = json.load(open(teacher)) if teacher else None
    split = {r["image_id"]: r["split"] for r in rows}
    train = [r for r in rows if r["split"] == "train"]
    tq = [q for r in train for q in r["questions"] if not is_aug(q)]  # train distinct
    allq = [q for r in rows for q in r["questions"]]
    vlm = [q for r in rows for q in r["questions"] if q["label_kind"] == "teacher_soft" and not is_aug(q)]
    results = []

    def check(name, target, value, ok, fmt="{:.3f}"):
        if value is None:
            results.append((name, target, "—", "MISSING"))
        else:
            v = fmt.format(value) if isinstance(value, float) else str(value)
            results.append((name, target, v, "PASS" if ok else "FAIL"))

    # --- structure
    bad = Counter(p for q in allq if (p := problem(q)))
    check("records pass merge.problem() validator", "0 invalid", sum(bad.values()), not bad)

    # --- per-image density (train)
    per = [sum(not is_aug(q) for q in r["questions"]) for r in train]
    check("train distinct questions / image", ">= 4.5", sum(per) / len(per), sum(per) / len(per) >= 4.5, "{:.2f}")
    f3 = sum(n >= 3 for n in per) / len(per)
    check("train images with >= 3 distinct questions", ">= 90%", f3, f3 >= 0.90, "{:.1%}")
    f2t = sum(len({q["type"] for q in r["questions"]}) >= 2 for r in train) / len(train)
    check("train images with >= 2 question types", ">= 75%", f2t, f2t >= 0.75, "{:.1%}")

    # --- mix (train distinct)
    mix = Counter(q["type"] for q in tq)
    # target mix 37.5 / 32.5 / 17.5 scaled to 100%, +-3 pp
    for t, lo, hi in (("bool", .40, .46), ("choice", .34, .40), ("score", .17, .23)):
        s = mix[t] / len(tq)
        check(f"train share {t}", f"{lo:.0%}-{hi:.0%}", s, lo <= s <= hi, "{:.1%}")
    nad = sum(q["na"] >= 0.5 for q in tq) / len(tq)
    check("train N/A-dominant (na >= 0.5)", "8%-12%", nad, 0.08 <= nad <= 0.12, "{:.1%}")

    # --- teacher bool balance
    vb = [q["target"]["yes"] / (q["target"]["yes"] + q["target"]["no"]) for q in vlm if q["type"] == "bool" and q["target"]]
    yr = sum(vb) / len(vb) if vb else None
    check("teacher bool mean P(yes)", "45%-55%", yr, yr is not None and 0.45 <= yr <= 0.55, "{:.1%}")

    # --- either/or bool: every regex hit must carry a classifier verdict that it's a plain yes/no question
    flagged = [q for q in allq if q["type"] == "bool" and EITHER_OR.search(q["q"])]
    unverified = sum(q.get("yesno_ok") is not True for q in flagged)
    check("either/or bool questions without a yes/no verdict", "0", unverified, unverified == 0)

    # --- N/A-like options anywhere
    na_opts = sum(any(NA_OPTION.search(o) or is_denial(o, q["q"]) for o in q["options"]) for q in allq if q["type"] == "choice")
    check("choice questions with an N/A-like option", "0", na_opts, na_opts == 0)

    # --- teacher option counts vs spec
    vc = Counter(len(q["options"]) for q in vlm if q["type"] == "choice")
    n = sum(vc.values())
    if n:
        dev_ = max(abs(vc[k] / n - p) for k, p in OPTION_SPEC.items())
        check("teacher choice option-count max deviation from spec", "<= 3 pp", dev_, dev_ <= 0.03, "{:.1%}")
    else:
        check("teacher choice option-count max deviation from spec", "<= 3 pp", None, False)

    # --- score scales
    vs = Counter(tuple(q["scale"]) for q in vlm if q["type"] == "score")
    ns = sum(vs.values())
    big = sum(c / ns >= 0.08 for c in vs.values()) if ns else None
    check("teacher score scales with >= 8% share", ">= 5", big, big is not None and big >= 5)

    # --- topics and phrasing
    topics = Counter(q.get("topic", q["family"]) for q in vlm)
    nt = sum(topics.values())
    check("teacher topics", ">= 25", len(topics), len(topics) >= 25)
    top = topics.most_common(1)[0][1] / nt if nt else None
    check("largest teacher topic share", "<= 8%", top, top is not None and top <= 0.08, "{:.1%}")
    op = Counter(" ".join(q["q"].lower().split()[:4]) for q in vlm)
    top_op = op.most_common(1)[0][1] / len(vlm) if vlm else None
    check("most common 4-word opening (teacher)", "<= 1%", top_op, top_op is not None and top_op <= 0.01, "{:.2%}")
    vo = [o for q in vlm if q["type"] == "choice" for o in q["options"]]
    up = sum(o[:1].isupper() for o in vo) / len(vo) if vo else None
    check("teacher options starting uppercase", "<= 5%", up, up is not None and up <= 0.05, "{:.1%}")

    # --- teacher calibration (from the bake-off report)
    for t in ("bool", "choice", "score"):
        e = cal and cal.get("eval", {}).get(t, {}).get("ece")
        check(f"teacher ECE after temperature ({t})", "<= 0.05", e, e is not None and e <= 0.05)
    auc = cal and cal.get("eval", {}).get("na", {}).get("auroc")
    check("teacher N/A AUROC", ">= 0.95", auc, auc is not None and auc >= 0.95)

    # --- position balance: argmax position within stored option order
    worst = None
    for k in range(2, 9):
        qs = [q for q in vlm if q["type"] == "choice" and len(q["options"]) == k and q["target"]]
        if len(qs) < 500:
            continue
        pos = Counter(q["options"].index(argmax(q["target"])) for q in qs)
        d = max(abs(pos[i] / len(qs) - 1 / k) for i in range(k))
        worst = d if worst is None else max(worst, d)
    check("teacher argmax position max deviation from uniform", "<= 3 pp", worst, worst is not None and worst <= 0.03, "{:.1%}")
    stab = [q["stability"] for q in vlm if "stability" in q]
    smax = max(stab) if stab else None
    lim = cal and cal.get("stability_threshold")
    check("teacher rotation instability (max JS) <= threshold", f"<= {lim}" if lim else "<= set in pilot", smax,
          smax is not None and lim is not None and smax <= lim)

    # --- held-out families
    for fam in sorted(HELD_OUT):
        c = sum(q["family"] == fam for r in rows if r["split"] == "test" for q in r["questions"])
        check(f"held-out test set '{fam}'", ">= 2500", c, c >= 2500)
    leak = sum(q["family"] in HELD_OUT for r in rows if r["split"] != "test" for q in r["questions"])
    check("held-out-family questions in train/val", "0", leak, leak == 0)

    # --- answer leaks in templates: the answer appears in the question text
    tl = sum(1 for q in allq if q["source"] in ("tpl_colour", "tpl_material") and q["target"]
             and re.search(rf"\b{re.escape(argmax(q['target']))}\b", q["q"].lower().replace("grey", "gray")))
    check("template questions containing their own answer", "0", tl, tl == 0)

    return results, {"file": path, "images": len(rows), "questions": len(allq), "train_distinct": len(tq),
                     "splits": dict(Counter(split.values()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--teacher", help="teacher calibration report JSON from the bake-off")
    ap.add_argument("--json", help="write results here (for report.py)")
    ap.add_argument("--only", help="jsonl with image_id per line: check only those images (pilot)")
    args = ap.parse_args()
    results, meta = run(args.dataset, args.teacher, args.only)
    print(f"{meta['file']}: {meta['images']:,} images, {meta['questions']:,} questions, "
          f"{meta['train_distinct']:,} distinct train questions, splits {meta['splits']}\n")
    w = max(len(r[0]) for r in results)
    for name, target, value, status in results:
        print(f"{status:8s} {name:{w}s}  target {target:>16s}  measured {value}")
    c = Counter(r[3] for r in results)
    print(f"\n{c['PASS']} PASS, {c['FAIL']} FAIL, {c['MISSING']} MISSING")
    if args.json:
        json.dump({"meta": meta, "results": results}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
