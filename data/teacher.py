"""Teacher pipeline: write questions from specs, label them with rotated options, calibrate against
human labels.

Writer: one request per spec (topics.py). The teacher sees the image, the image's existing questions,
the spec (kind, exact option count / scale, topic + 2 alternates, intended answer for bool, false
premise for N/A specs) and 3 style exemplars; guided JSON fixes the shape. Dedup is per image only
(v1's run-wide dedup silently dropped any question once asked about an earlier image).

Labels: one prompt per (question, rotation). Options are lettered A, B, ...; an explicit N/A option
("cannot be answered from this image ...") is one of the letters and rotates like the others, so no
answer or N/A ever sits in a fixed position. The first generated token is restricted to the offered
letters and read as a distribution. Rotations are stored raw; finalize applies the per-type temperature
and N/A bias fitted on human labels, averages rotations, and drops questions whose rotations disagree.

    CPU:  demo | calibset | finalize
    GPU (vLLM 0.29.0 + a 32B VLM): preflight | calibrate | write | classify | label

All randomness is a hash of the image id; with VLLM_BATCH_INVARIANT=1, temperature-0 readouts and seeded
sampling, re-running a shard reproduces it byte for byte.
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import OUT, family as keyword_family, read_jsonl, write_jsonl
from merge import HELD_OUT, is_denial
from merge import EITHER_OR
from qa import NA_OPTION
from topics import TOPICS, h

LETTERS = "ABCDEFGHIJKL"  # up to 11 scale values (0-10) + N/A
NA_TEXTS = {
    "premise": "cannot be answered from this image (what it asks about is not shown, or its premise is false)",
    "fit": "not applicable: the question doesn't fit this image",
}
NA = "<N/A>"  # label key for the N/A option inside distributions
LOG_FLOOR = -100.0  # log-probability for a letter vLLM didn't return (should not happen, see readout)


# ----------------------------------------------------------------------------------------------
# questions -> labelling prompts (pure)

def items(q):
    """Answer labels in canonical order, N/A last. Choice options are shuffled by a hash first, so the
    order the writer (or a human dataset) listed them in can't leak into the label."""
    if q["type"] == "bool":
        return ["yes", "no", NA]
    if q["type"] == "choice":
        opts = sorted(q["options"], key=lambda o: h(q.get("image_id", ""), q["q"], o))
        return opts + [NA]
    lo, hi = q["scale"]
    return [str(v) for v in range(lo, hi + 1)] + [NA]


def rotations(q):
    """Orders to present the labels in. bool: all 3 cyclic shifts (every label in every position once).
    choice: min(n, 4) evenly spaced cyclic shifts. score: ascending with N/A last, descending with N/A first."""
    base = items(q)
    n = len(base)
    if q["type"] == "score":
        return [base, [NA] + base[:-1][::-1]]
    r = n if q["type"] == "bool" else min(n, 4)
    return [base[s:] + base[:s] for s in sorted({round(i * n / r) for i in range(r)})]


def option_text(q, label, na_text):
    if label == NA:
        return na_text
    if q["type"] == "score":
        lo, hi = q["scale"]
        a_lo, a_hi = q["anchors"]
        return f"{label} ({a_lo})" if label == str(lo) else f"{label} ({a_hi})" if label == str(hi) else label
    return label


def label_prompt(q, order, na_text):
    lines = [q["q"]]
    if q["type"] == "score":
        lo, hi = q["scale"]
        lines.append(f"Rate on a scale from {lo} ({q['anchors'][0]}) to {hi} ({q['anchors'][1]}).")
    lines += [f"{LETTERS[i]}. {option_text(q, lab, na_text)}" for i, lab in enumerate(order)]
    lines.append("Answer with the letter of the best option.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# readouts -> distributions (pure)

def readout(logprobs, n, letter_ids):
    """Letter-token logprobs (vLLM's {token_id: Logprob}) -> log-probabilities over the n offered letters,
    renormalised (log-softmax), kept in log space: storing probabilities rounded to 1e-6 capped every
    log-ratio at ~13.8, and at Qwen3-VL's fitted T~3.5 that floor became ~2% per option (calibration run
    1). A letter missing from the returned top-k gets LOG_FLOOR; with logprobs_mode=
    "processed_logprobs" (load_llm) the masked distribution is returned, so none should be missing."""
    raw = []
    for i in range(n):
        lp = logprobs.get(letter_ids[LETTERS[i]])
        v = lp.logprob if hasattr(lp, "logprob") else lp
        raw.append(LOG_FLOOR if v is None or not math.isfinite(v) else max(LOG_FLOOR, v))
    m = max(raw)
    z = m + math.log(sum(math.exp(x - m) for x in raw))
    return [x - z for x in raw]


def letter_bias(qs, iters=25, clip=-20.0):
    """The teacher's preference for answer letters, per number of offered letters n, in log space (PriDe,
    Zheng et al., ICLR 2024). Choice questions with n > 4 labels get only 4 of the n cyclic shifts, so each
    option visits 4 letters and a letter preference doesn't cancel in the average (full run: at k = 7 the last
    option won 22% instead of 14%). Model, after centring each rotation's lp (which removes its normaliser):
    lp[q, r, item] = a[q, item] + b[n][letter]; fitted by alternating means, lp clipped at `clip` so the
    teacher's -30 runner-ups don't dominate. Returns {n: [b_0 .. b_n-1]}, each centred to mean 0."""
    groups = defaultdict(list)  # n -> [(item group, letter, centred lp)]
    g = 0
    for q in qs:
        idx = {}
        for r in q["rots"]:
            v = [max(x, clip) for x in r["lp"]]
            m = sum(v) / len(v)
            for j, (lab, x) in enumerate(zip(r["order"], v)):
                key = idx.setdefault(lab, g + len(idx))
                groups[len(v)].append((key, j, x - m))
        g += len(idx)
    out = {}
    for n, rows in groups.items():
        b = [0.0] * n
        for _ in range(iters):
            sa, ca = defaultdict(float), defaultdict(int)
            for key, j, v in rows:
                sa[key] += v - b[j]; ca[key] += 1
            sb, cb = [0.0] * n, [0] * n
            for key, j, v in rows:
                sb[j] += v - sa[key] / ca[key]; cb[j] += 1
            b = [s / c if c else 0.0 for s, c in zip(sb, cb)]
            mean = sum(b) / n
            b = [x - mean for x in b]
        out[n] = [round(x, 4) for x in b]
    return out


def calibrated(lp, order, T=1.0, na_bias=0.0, lb=None):
    """Remove the letter preference `lb` (by letter index, see letter_bias), then temperature T on every
    option's log-probability and an additive logit bias on N/A, then renormalise."""
    logits = [(x - (lb[j] if lb else 0.0)) / T + (na_bias if lab == NA else 0.0)
              for j, (x, lab) in enumerate(zip(lp, order))]
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    z = sum(e)
    return {lab: x / z for lab, x in zip(order, e)}


def js(p, q):
    def kl(a, b):
        return sum(a[k] * math.log(a[k] / b[k]) for k in a if a[k] > 0)
    m = {k: (p[k] + q[k]) / 2 for k in p}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def aggregate(rots, T=1.0, na_bias=0.0, lb=None):
    """rots: [{"order": [...], "lp": [...]}]. lb: {n letters: letter bias} or None. Returns (mean
    distribution over labels, stability = max JS divergence of any rotation from the mean; 0 = every
    rotation agrees)."""
    lb = {int(k): v for k, v in (lb or {}).items()}  # JSON keys are strings
    dists = [calibrated(r["lp"], r["order"], T, na_bias, lb.get(len(r["lp"]))) for r in rots]
    mean = {k: sum(d[k] for d in dists) / len(dists) for k in dists[0]}
    return mean, max(js(d, mean) for d in dists)


def regime(na):
    """index_prior is estimated separately for answerable and N/A-dominant questions: when the teacher says
    the question can't be answered, the leftover spread over the options carries little signal and the
    positional artefact dominates (full run: the option next to N/A won 36-66% instead of 1/k)."""
    return "na" if na >= 0.5 else "answerable"


def to_target(q, dist, ip=None):
    """(target, na) in the project's record format: target is a distribution over the real answers
    summing to 1 (independent of na), or None when na is 1. ip: {k options: prior per position} for choice
    questions (index_prior): each option's probability is divided by its position's prior. `dist` keeps
    rotation 0's order, i.e. items(q) order, so position i of `rest` is option position i."""
    na = dist[NA]
    rest = {k: v for k, v in dist.items() if k != NA}
    prior = (ip or {}).get(regime(na), {}).get(str(len(rest))) if q["type"] == "choice" else None
    if prior:
        rest = {k: v / p for (k, v), p in zip(rest.items(), prior)}
    z = sum(rest.values())
    if na >= 1.0 or z <= 0:
        return None, 1.0
    if q["type"] == "score":
        lo, hi = q["scale"]
        return [rest[str(v)] / z for v in range(lo, hi + 1)], na
    return {k: v / z for k, v in rest.items()}, na


# ----------------------------------------------------------------------------------------------
# calibration metrics (pure)

def q_loss(q_true, target, na):
    """Training-style loss: BCE on N/A + (1 - true N/A) * cross-entropy of the answer distribution."""
    eps = 1e-9
    t_na = q_true["na"]
    loss = -(t_na * math.log(max(na, eps)) + (1 - t_na) * math.log(max(1 - na, eps)))
    if q_true["target"] is None or target is None:
        return loss
    t = q_true["target"]
    pairs = zip(t, target) if isinstance(t, list) else ((t[k], target.get(k, 0)) for k in t)
    return loss + (1 - t_na) * -sum(a * math.log(max(b, eps)) for a, b in pairs)


def ece(pairs, bins=15):
    """pairs: (confidence, accuracy) with accuracy in [0, 1] (soft: the human mass on the predicted answer)."""
    by = defaultdict(list)
    for c, a in pairs:
        by[min(int(c * bins), bins - 1)].append((c, a))
    n = len(pairs)
    return sum(len(v) / n * abs(sum(c for c, _ in v) / len(v) - sum(a for _, a in v) / len(v)) for v in by.values())


def auroc(scores_labels):
    pos = sorted(s for s, y in scores_labels if y)
    neg = sorted(s for s, y in scores_labels if not y)
    if not pos or not neg:
        return None
    # Mann-Whitney U with ties counted half
    import bisect
    u = sum(bisect.bisect_left(neg, s) + 0.5 * (bisect.bisect_right(neg, s) - bisect.bisect_left(neg, s)) for s in pos)
    return u / (len(pos) * len(neg))


def index_prior(qs, T, na_bias, lb=None):
    """Average answer probability per option position, per option count k, over choice questions (PriDe's
    prior over option IDs). Option order is an answer-independent hash shuffle, so every position should
    average 1/k; what's left is a positional artefact of the teacher. Full run: with the letter preference
    removed, the option listed just before the rotating N/A option (cyclically adjacent in every rotation)
    still won 23% instead of 14% at k = 7. Returns {regime: {k: [prior_0 .. prior_k-1]}}, each summing to 1."""
    sums, n = defaultdict(lambda: defaultdict(float)), Counter()
    for q in qs:
        if q["type"] != "choice":
            continue
        tg, na = to_target(q, aggregate(q["rots"], T.get("choice", 1.0), na_bias.get("choice", 0.0), (lb or {}).get("choice"))[0])
        if tg:
            key = (regime(na), len(tg))
            for i, v in enumerate(tg.values()):
                sums[key][i] += v
            n[key] += 1
    out = defaultdict(dict)
    for (r, k) in sorted(n):
        out[r][str(k)] = [round(sums[(r, k)][i] / n[(r, k)], 5) for i in range(k)]
    return dict(out)


def evaluate(rows, T, na_bias, lb=None, ip=None):
    """rows: calibration questions with "rots". Per-type loss, ECE, accuracy; N/A AUROC."""
    out, na_pairs = {}, []
    by_type = defaultdict(list)
    for q in rows:
        dist, stab = aggregate(q["rots"], T.get(q["type"], 1.0), na_bias.get(q["type"], 0.0), (lb or {}).get(q["type"]))
        target, na = to_target(q, dist, (ip or {}).get(q["type"]))
        na_pairs.append((na, q["na"] >= 0.5))
        by_type[q["type"]].append((q, target, na, stab))
    for t, rs in by_type.items():
        ans = [(q, tg) for q, tg, _, _ in rs if q["target"] is not None and tg is not None and q["na"] < 0.5]
        conf, acc = [], []
        for q, tg in ans:
            if isinstance(tg, list):
                k = max(range(len(tg)), key=tg.__getitem__)
                conf.append(tg[k]); acc.append(q["target"][k])
            else:
                k = max(tg, key=tg.get)
                conf.append(tg[k]); acc.append(q["target"].get(k, 0.0))
        out[t] = {"n": len(rs), "loss": sum(q_loss(q, tg, na) for q, tg, na, _ in rs) / len(rs),
                  "ece": ece(list(zip(conf, acc))) if conf else None,
                  "acc": sum(acc) / len(acc) if acc else None,
                  "stability_p95": sorted(s for *_, s in rs)[int(0.95 * len(rs))]}
    out["na"] = {"auroc": auroc(na_pairs), "n_pos": sum(y for _, y in na_pairs)}
    return out


def fit(rows, lb=None):
    """Per-type temperature and N/A bias minimising the training loss (grid search; few parameters)."""
    T, b = {}, {}
    Ts = [round(0.3 * 1.1 ** i, 4) for i in range(48)]  # 0.3 .. ~26: Qwen3-VL's letter logits are very peaked
    bs = [x / 4 for x in range(-16, 17)]  # -4 .. 4
    for t in ("bool", "choice", "score"):
        rs = [q for q in rows if q["type"] == t]
        if not rs:
            continue

        def loss(Tv, bv):
            total = 0.0
            for q in rs:
                tg, na = to_target(q, aggregate(q["rots"], Tv, bv, (lb or {}).get(t))[0])
                total += q_loss(q, tg, na)
            return total / len(rs)
        best_T = min(Ts, key=lambda Tv: loss(Tv, 0.0))
        best_b = min(bs, key=lambda bv: loss(best_T, bv)) if any(q["na"] >= 0.5 for q in rs) else 0.0
        best_T = min(Ts, key=lambda Tv: loss(Tv, best_b))
        T[t], b[t] = best_T, best_b
    return T, b


# ----------------------------------------------------------------------------------------------
# writer prompts, schemas and post-processing (pure)

KIND_TEXT = {
    "bool_yes": "a yes/no question whose correct answer for this image is YES",
    "bool_no": "a yes/no question whose correct answer for this image is NO: ask about something plausible for "
               "this kind of scene that is clearly false here (an object that is absent, a wrong colour, a wrong "
               "count, an action that is not happening)",
    "choice": "a multiple-choice question with exactly {k} options, exactly one of which fits this image best. "
              "It must be a what/which/where/how/who question that the options answer directly -- never a "
              "yes/no question, and never \"yes\" or \"no\" as options",
    "score": "a question asking to rate a property on a scale from {lo} to {hi}, with short labels for what "
             "{lo} and {hi} mean. Pick a property that varies a lot between images, so the answer for this "
             "image could be anywhere on the scale, including either end",
    "na_bool": "a yes/no question with a FALSE PREMISE. First name, in absent_thing, a specific object that would "
               "be plausible in this kind of scene but is NOT in this image. Then ask a yes/no question that "
               "takes that object for granted and asks about one of its properties or actions (e.g. absent_thing "
               "\"dog\": \"Is the dog asleep?\" or \"Is the dog's collar red?\"). Never ask whether it is there "
               "(\"Is there a dog?\" has a real answer, no -- that is not a false premise)",
    "na_choice": "a multiple-choice question with exactly {k} options and a FALSE PREMISE. First name, in "
                 "absent_thing, a specific object that would be plausible in this kind of scene but is NOT in "
                 "this image. Then ask a what/which/where/how question that takes that object for granted "
                 "(e.g. absent_thing \"dog\": \"What colour is the dog's collar?\"), with options that would "
                 "answer it if the object were there. Never \"yes\" or \"no\" as options",
}
# The question's first word, enforced while decoding (guided JSON "pattern"). Pilot 1 (2026-10-01): 451 of
# 991 choice questions were yes/no questions with yes/no/maybe options, despite the prompt.
CHOICE_START = r"^(What|Which|Where|How|Who|Whose|When|Why|In|On|At|From|Approximately|Roughly) "
BOOL_START = r"^(Is|Are|Was|Were|Does|Do|Did|Can|Could|Has|Have|Had|Will|Would|Should|May|Might) "
RULES = ("Rules: the question must be about this specific image. Don't reveal the answer in the question. "
         "Don't start with \"What is the\" and don't use the phrase \"in the image\". A yes/no question must "
         "be answerable with a plain yes or no -- never \"Is it X or Y?\" (that is a multiple-choice question). "
         "Options are short (1-4 words), lowercase, mutually exclusive, and never \"cannot determine\", "
         "\"unknown\", \"unclear\" or \"none of the above\" (that case is handled separately).")


def writer_prompt(spec, existing, split):
    k, (lo, hi) = spec.get("k"), spec.get("scale", (0, 0))
    topics = "; ".join(f"{t} ({TOPICS[t][1]})" for t in spec["topics"])
    lines = ["Write one new question about this image for a visual question-answering dataset.",
             "Write " + KIND_TEXT[spec["kind"]].format(k=k, lo=lo, hi=hi) + ".",
             f"Topic: choose the first of these that fits this image: {topics}."
             if len(spec["topics"]) > 1 else f"Topic: {topics}.",
             RULES]
    if split != "test" and not spec.get("held_out"):
        lines.append("Don't ask what anything is made of, and don't ask whether anything is open or closed.")
    if existing:
        lines.append("Existing questions about this image (don't repeat or paraphrase them): "
                     + " | ".join(existing[:12]))
    lines.append("Phrasing examples from other images (style only, don't copy): " + " | ".join(spec["exemplars"]))
    lines.append("If no listed topic can be asked about this image in the required form, set skip to true.")
    return "\n".join(lines)


def writer_schema(spec):
    props = {"topic": {"enum": spec["topics"]}}
    if spec["kind"].startswith("na_"):  # commit to the absent object before writing the question (pilot 1: na_bool
        props["absent_thing"] = {"type": "string", "maxLength": 40}  # specs came out as ordinary questions)
    props["question"] = {"type": "string", "maxLength": 200}
    start = {"choice": CHOICE_START, "bool": BOOL_START}.get(spec["type"])
    if start:
        props["question"]["pattern"] = start + r"[^?]*\?$"
    if spec["type"] == "choice":
        props["options"] = {"type": "array", "items": {"type": "string", "maxLength": 50},
                            "minItems": spec["k"], "maxItems": spec["k"]}
    if spec["type"] == "score":
        props["low_label"] = {"type": "string", "maxLength": 40}
        props["high_label"] = {"type": "string", "maxLength": 40}
    props["skip"] = {"type": "boolean"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def norm_words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def near_dup(a, b):
    wa, wb = set(norm_words(a)), set(norm_words(b))
    return bool(wa) and bool(wb) and len(wa & wb) / len(wa | wb) >= 0.8


def postprocess(spec, raw, image_id, split, existing):
    """Writer JSON -> question dict, or (None, reason). Deterministic."""
    try:
        o = json.loads(raw)
    except json.JSONDecodeError:
        return None, "bad json"
    if o.get("skip"):
        return None, "skipped"
    q = " ".join(str(o.get("question", "")).split())
    q = q[:1].upper() + q[1:]  # human questions are capitalised; a lowercase start would be a source cue
    if spec["type"] != "score":
        q = q.rstrip(" .") + ("" if q.endswith("?") else "?")  # pilot 1: 126 lost to a missing "?"
        start = CHOICE_START if spec["type"] == "choice" else BOOL_START
        if not re.match(start, q):  # backstop for the decoding pattern
            return None, "wrong question form"
    if not 12 <= len(q) <= 200:
        return None, "length"
    topic = o.get("topic") if o.get("topic") in spec["topics"] else spec["topics"][0]
    fam = TOPICS[topic][0]
    rec = {"image_id": image_id, "slot": spec["slot"], "kind": spec["kind"], "type": spec["type"], "topic": topic,
           "family": fam, "q": q}
    if spec["type"] == "choice":
        opts = [" ".join(str(x).split()).lower().strip(" .") for x in o.get("options", [])]
        if len(opts) != spec["k"] or len({" ".join(norm_words(x)) for x in opts}) != len(opts) or not all(opts):
            return None, "options not distinct"
        if any(NA_OPTION.search(x) or is_denial(x, q) or len(x.split()) > 6 for x in opts):
            return None, "N/A-like or long option"
        if {"yes", "no", "maybe", "uncertain"} & set(opts):
            return None, "yes/no or hedge option"
        words = set(norm_words(q))
        # an option fully contained in the question ("Which colour is the red kite?" / red) gives it away;
        # 1-2 letter options ("no", "up") are too common to count
        if any(len(x) >= 3 and set(norm_words(x)) <= words for x in opts):
            return None, "answer leak"
        rec["options"] = opts
    if spec["type"] == "score":
        lo, hi = spec["scale"]
        a = [" ".join(str(o.get(f, "")).split()).lower() for f in ("low_label", "high_label")]
        if not all(a) or a[0] == a[1]:
            return None, "bad anchors"
        rec["scale"], rec["anchors"] = [lo, hi], a
    if spec["kind"].startswith("na_"):
        thing = norm_words(str(o.get("absent_thing", "")))
        if not thing or thing[-1] not in {w.rstrip("s") for w in norm_words(q)} | set(norm_words(q)):
            return None, "N/A question doesn't presuppose its absent object"
        if re.match(r"^(Is|Are) there\b|^(Can|Do|Does) (you|we|one) see\b", q):
            return None, "presence question on an N/A spec"
        rec["premise"] = " ".join(thing)
    held = spec.get("held_out", False)
    if not held and (fam in HELD_OUT or keyword_family(q) in HELD_OUT):
        return None, "held-out family on a non-test spec"
    if held:
        rec["held_out"] = True
    if any(near_dup(q, e) for e in existing):
        return None, "near-duplicate of an existing question"
    if spec["type"] == "bool" and EITHER_OR.search(q):
        rec["check_yesno"] = True
    return rec, None


# ----------------------------------------------------------------------------------------------
# calibration set from human labels (CPU)

CALIB = {  # source -> how many questions; train+val images only (test stays untouched)
    "vqav2": 1200, "gqa": 600, "tpl_presence": 600, "tpl_relation": 300,          # bool, answerable
    "vqav2_choice": 1200, "aokvqa": 900, "tpl_colour": 600,                         # choice
    "koniq": 1200, "tallyqa": 1200,                                                 # score
    "tpl_na": 800, "vizwiz": 250,                                                   # N/A positives
}


def calibset():
    rows = []
    pools = defaultdict(list)
    for r in read_jsonl(OUT / "v2/base_selected.jsonl"):
        if r["split"] == "test":
            continue
        for q in r["questions"]:
            if q.get("aug"):
                continue
            src = "tallyqa" if q["source"].startswith("tallyqa") else q["source"]
            if src in CALIB:
                pools[src].append({**q, "image_id": r["image_id"]})
    for src, n in sorted(CALIB.items()):
        pool = sorted(pools[src], key=lambda q: h("calib", q["image_id"], q["q"]))[:n]
        for q in pool:
            q["half"] = "fit" if h("half", q["image_id"]) % 2 else "eval"
        rows += pool
        print(f"{src}: {len(pool)}")
    write_jsonl(OUT / "v2/calib.jsonl", rows)


# ----------------------------------------------------------------------------------------------
# vLLM (GPU) parts

def load_llm(model, seed=0):
    import os

    from vllm import LLM
    print("VLLM_BATCH_INVARIANT =", os.environ.get("VLLM_BATCH_INVARIANT"), flush=True)
    # processed_logprobs: logprobs after allowed_token_ids masking, i.e. already over the offered letters
    return LLM(model=model, limit_mm_per_prompt={"image": 1}, max_model_len=8192, seed=seed,
               enable_prefix_caching=True, logprobs_mode="processed_logprobs")


def chat(tok, text, image=True):
    content = ([{"type": "image"}] if image else []) + [{"type": "text", "text": text}]
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)


def letter_ids(tok):
    ids = {L: tok.convert_tokens_to_ids(L) for L in LETTERS}
    assert all(isinstance(i, int) and i >= 0 for i in ids.values()), ids
    return ids


def label_requests(qs, imgs, tok, na_text):
    """One request per (question, rotation); requests for one image are adjacent so prefix caching can
    reuse the image."""
    from vllm import SamplingParams

    lids = letter_ids(tok)
    reqs, params, meta = [], [], []
    for qi, q in enumerate(qs):
        for order in rotations(q):
            reqs.append({"prompt": chat(tok, label_prompt(q, order, na_text)),
                         "multi_modal_data": {"image": imgs[q["image_id"]]}})
            params.append(SamplingParams(max_tokens=1, temperature=0.0, logprobs=20,
                                         allowed_token_ids=[lids[LETTERS[i]] for i in range(len(order))]))
            meta.append((qi, order))
    return reqs, params, meta, lids


def run_labels(llm, tok, qs, imgs, na_text):
    reqs, params, meta, lids = label_requests(qs, imgs, tok, na_text)
    outs = llm.generate(reqs, params)
    rots, missing = defaultdict(list), 0
    for (qi, order), o in zip(meta, outs):
        lp = o.outputs[0].logprobs[0]
        # processed (masked) logprobs: every offered letter should be returned, finite; count any that aren't
        missing += any(lids[LETTERS[i]] not in lp or not math.isfinite(lp[lids[LETTERS[i]]].logprob)
                       for i in range(len(order)))
        rots[qi].append({"order": order, "lp": [round(x, 4) for x in readout(lp, len(order), lids)]})
    print(f"rotations with an offered letter missing from the returned logprobs: {missing}/{len(outs)}", flush=True)
    return [rots[i] for i in range(len(qs))]


def open_images(ids, images_dir):
    from PIL import Image
    out = {}
    for i in ids:
        try:
            out[i] = Image.open(Path(images_dir) / f"{i}.jpg").convert("RGB")
        except Exception as e:  # dead/corrupt file: questions on it are skipped, not a crash
            print(f"  skipping image {i}: {e}", flush=True)
    # a few dead files are fine; most missing means a wrong --images dir (calibration run 2 lost a job to that)
    assert len(out) >= 0.95 * len(ids), f"only {len(out)}/{len(ids)} images found in {images_dir}"
    return out


def chunked(rows, out_path, chunk, fn, key="image_id"):
    """Resumable: rows already in out_path (by key) are skipped; each chunk's results are appended at once."""
    out_path = Path(out_path)
    done = {r[key] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r[key] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    with open(out_path, "a") as f:
        for s in range(0, len(todo), chunk):
            for r in fn(todo[s:s + chunk]):
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            print(f"  {min(s + chunk, len(todo))}/{len(todo)}", flush=True)


def cmd_calibrate(args):
    llm = load_llm(args.model)
    tok = llm.get_tokenizer()
    rows = list(read_jsonl(Path(args.calib)))
    imgs = open_images(sorted({q["image_id"] for q in rows}), args.images)
    rows = [q for q in rows if q["image_id"] in imgs]
    import time
    report = {"model": args.model, "na_texts": {}}
    for name in args.na_texts:
        text = NA_TEXTS[name]
        t0 = time.time()
        rots = run_labels(llm, tok, rows, imgs, text)
        dt = time.time() - t0
        for q, r in zip(rows, rots):
            q["rots"] = r
        fit_rows = [q for q in rows if q["half"] == "fit"]
        eval_rows = [q for q in rows if q["half"] == "eval"]
        T, b = fit(fit_rows)
        n_req = sum(len(r) for r in rots)
        report["na_texts"][name] = {
            "T": T, "na_bias": b, "requests": n_req, "seconds": dt, "req_per_s": n_req / dt,
            "eval_raw": evaluate(eval_rows, {}, {}), "eval": evaluate(eval_rows, T, b),
            "position_share": position_share(eval_rows)}
        json.dump(rots, open(Path(args.out).with_suffix(f".{name}.rots.json"), "w"))
        print(name, json.dumps(report["na_texts"][name], indent=1), flush=True)
    best = min(report["na_texts"], key=lambda k: sum(v["loss"] for t, v in report["na_texts"][k]["eval"].items() if t != "na"))
    report["chosen_na_text"] = best
    report.update({k: report["na_texts"][best][k] for k in ("T", "na_bias", "eval")})
    json.dump(report, open(args.out, "w"), indent=1)
    print("chosen N/A wording:", best)


def cmd_recalibrate(args):
    """CPU: add a letter-bias correction (letter_bias) for --types, estimated from a full labelled run, and
    refit T / N/A bias on the saved calibration readouts with it applied. Writes a new calibration file."""
    cal = json.load(open(args.calibration))
    rots = json.load(open(args.rots))
    skip = set(open(args.skip).read().split()) if args.skip else set()
    rows = [q for q in read_jsonl(Path(args.calib)) if q["image_id"] not in skip]
    assert len(rows) == len(rots), (len(rows), len(rots))
    for q, r in zip(rows, rots):
        q["rots"] = r
    lab = [q for r in read_jsonl(Path(args.labelled)) for q in r["questions"] if q["type"] in args.types]
    lb = {t: letter_bias([q for q in lab if q["type"] == t]) for t in args.types}
    fit_rows = [q for q in rows if q["half"] == "fit"]
    eval_rows = [q for q in rows if q["half"] == "eval"]
    T, b = fit(fit_rows, lb)
    ip = {"choice": index_prior(lab, T, b, lb)} if "choice" in args.types else {}
    print("index prior:", json.dumps(ip))
    print("letter correction only:", {t: round(v["loss"], 4) for t, v in evaluate(eval_rows, T, b, lb).items() if t != "na"})
    before, after = cal["eval"], evaluate(eval_rows, T, b, lb, ip)
    for t in ("bool", "choice", "score"):
        print(f"{t:6s} before: loss {before[t]['loss']:.4f} ece {before[t]['ece']:.4f} acc {before[t]['acc']:.4f} | "
              f"after: loss {after[t]['loss']:.4f} ece {after[t]['ece']:.4f} acc {after[t]['acc']:.4f}  (T {T[t]}, N/A bias {b[t]})")
    print("N/A AUROC", round(before["na"]["auroc"], 4), "->", round(after["na"]["auroc"], 4))
    print("letter bias:", json.dumps(lb))
    json.dump({**cal, "T": T, "na_bias": b, "eval": after, "letter_bias": lb, "index_prior": ip,
               "recalibrated_from": {"calibration": args.calibration, "labelled": args.labelled}},
              open(args.out, "w"), indent=1)


def position_share(rows):
    """Share of questions whose raw (uncalibrated) argmax letter is in each position, per rotation."""
    pos = Counter()
    n = 0
    for q in rows:
        for r in q["rots"]:
            pos[max(range(len(r["lp"])), key=r["lp"].__getitem__)] += 1
            n += 1
    return {i: round(c / n, 4) for i, c in sorted(pos.items())}


def cmd_write(args):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    llm = load_llm(args.model)
    tok = llm.get_tokenizer()
    specs = select_rows(read_jsonl(Path(args.specs)), args)

    def fn(rows):
        imgs = open_images([r["image_id"] for r in rows], args.images)
        reqs, params, meta = [], [], []
        for r in rows:
            if r["image_id"] not in imgs:
                continue
            for s in r["specs"]:
                reqs.append({"prompt": chat(tok, writer_prompt(s, r["existing"], r["split"])),
                             "multi_modal_data": {"image": imgs[r["image_id"]]}})
                params.append(SamplingParams(max_tokens=300, temperature=0.7, top_p=0.95,
                                             seed=h(r["image_id"], s["slot"]) % 2**31,
                                             structured_outputs=StructuredOutputsParams(json=writer_schema(s))))
                meta.append((r, s))
        outs = llm.generate(reqs, params)
        per = defaultdict(lambda: {"questions": [], "rejected": []})
        for (r, s), o in zip(meta, outs):
            existing = r["existing"] + [q["q"] for q in per[r["image_id"]]["questions"]]
            rec, why = postprocess(s, o.outputs[0].text, r["image_id"], r["split"], existing)
            if rec:
                per[r["image_id"]]["questions"].append(rec)
            else:
                per[r["image_id"]]["rejected"].append({"slot": s["slot"], "kind": s["kind"], "reason": why,
                                                       "raw": o.outputs[0].text[:400]})
        return [{"image_id": r["image_id"], "split": r["split"], **per[r["image_id"]]} for r in rows]

    chunked(specs, args.out, args.chunk, fn)


YESNO_PROMPT = ("Question: \"{q}\"\nCan this question be fully answered with just \"yes\" or \"no\"?\n"
                "{A}. {a}\n{B}. {b}\nAnswer with the letter.")
YESNO_OPTS = ("yes -- it asks whether something is true", "no -- it asks which of two or more alternatives is the case")


def cmd_classify(args):
    """Either/or check (text only, no image): every bool question matching EITHER_OR, from the teacher's
    written questions and the human sources. Two rotations of A/B. Either/or questions are dropped: teacher
    ones in `label`, human ones in merge. Verdicts -> out_path (one row per question)."""
    from vllm import SamplingParams
    llm = load_llm(args.model)
    tok = llm.get_tokenizer()
    lids = letter_ids(tok)
    todo = []
    for path in args.inputs:
        for r in read_jsonl(Path(path)):
            for q in r["questions"]:
                if q["type"] == "bool" and EITHER_OR.search(q["q"]):
                    todo.append({"image_id": r["image_id"], "q": q["q"], "source": q.get("source", "teacher")})
    reqs, params = [], []
    for t in todo:
        for flip in (False, True):
            a, b = YESNO_OPTS[::-1] if flip else YESNO_OPTS
            reqs.append({"prompt": chat(tok, YESNO_PROMPT.format(q=t["q"], A="A", B="B", a=a, b=b), image=False)})
            params.append(SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, allowed_token_ids=[lids["A"], lids["B"]]))
    outs = llm.generate(reqs, params)
    for i, t in enumerate(todo):
        pa = math.exp(readout(outs[2 * i].outputs[0].logprobs[0], 2, lids)[0])      # P(plain yes/no), A first
        pb = math.exp(readout(outs[2 * i + 1].outputs[0].logprobs[0], 2, lids)[1])  # same, presented second
        t["p_yesno"] = round((pa + pb) / 2, 6)
        t["yesno_ok"] = t["p_yesno"] >= 0.5
    write_jsonl(Path(args.out), todo)
    print(f"{len(todo)} checked, {sum(not t['yesno_ok'] for t in todo)} either/or", flush=True)


def select_rows(rows, args):
    """Rows in hash order (so --limit N is a representative sample: the pilot is the first 1,000 of the
    full run, not the first 1,000 ids alphabetically), then --shard i/n for parallel GPUs."""
    rows = sorted(rows, key=lambda r: h("order", r["image_id"]))
    if args.limit:
        rows = rows[:args.limit]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        rows = rows[i::n]
    return rows


def cmd_label(args):
    llm = load_llm(args.model)
    tok = llm.get_tokenizer()
    cal = json.load(open(args.calibration))
    na_text = NA_TEXTS[cal["chosen_na_text"]]
    rows = select_rows(read_jsonl(Path(args.inputs)), args)
    verdicts = {(v["image_id"], v["q"]): v for v in read_jsonl(Path(args.verdicts))} if args.verdicts else {}

    def fn(chunk):
        imgs = open_images([r["image_id"] for r in chunk], args.images)
        qs = []
        for r in chunk:
            for q in r["questions"]:
                q = {**q, "image_id": r["image_id"]}
                v = verdicts.get((r["image_id"], q["q"]))
                if v is not None and not v["yesno_ok"]:
                    # dropped, not converted to a 2-option choice: in pilot 3 one of the two flagged was a real
                    # either/or ("grazing or being led?") and one inclusive ("chipped or damaged?"), so a
                    # conversion is wrong about half the time; dropping costs ~0.2% of bools
                    continue
                elif v is not None:
                    q["yesno_ok"] = True
                # hybrid teacher (Gate 1): one model per question type
                if r["image_id"] in imgs and q["type"] in args.types:
                    qs.append(q)
        rots = run_labels(llm, tok, qs, imgs, na_text)
        per = defaultdict(list)
        for q, rr in zip(qs, rots):
            per[q["image_id"]].append({**{k: v for k, v in q.items() if k != "image_id"}, "rots": rr})
        return [{"image_id": r["image_id"], "split": r.get("split"), "questions": per[r["image_id"]]} for r in chunk]

    chunked(rows, args.out, args.chunk, fn)


def cmd_preflight(args):
    """Cheap API check before spending GPU time: letter tokens, restricted-first-token logprobs, per-spec
    guided JSON, seeded sampling reproducibility."""
    from PIL import Image
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    llm = load_llm(args.model)
    tok = llm.get_tokenizer()
    lids = letter_ids(tok)
    print("letter ids:", lids)
    img = Image.new("RGB", (448, 336), (40, 120, 200))
    q = {"image_id": "x", "type": "choice", "q": "What colour is the picture?", "options": ["blue", "red", "green"]}
    for order in rotations(q):
        out = llm.generate([{"prompt": chat(tok, label_prompt(q, order, NA_TEXTS["premise"])), "multi_modal_data": {"image": img}}],
                           SamplingParams(max_tokens=1, temperature=0.0, logprobs=20,
                                          allowed_token_ids=[lids[LETTERS[i]] for i in range(len(order))]))
        lp = out[0].outputs[0].logprobs[0]
        print(order, "returned:", sorted((v.decoded_token, round(v.logprob, 3)) for v in lp.values())[:6],
              "->", [round(math.exp(x), 3) for x in readout(lp, len(order), lids)])
    spec = {"slot": 0, "kind": "choice", "type": "choice", "k": 5, "topics": ["colour", "lighting", "composition"],
            "exemplars": ["How bright is it?"]}
    sp = SamplingParams(max_tokens=300, temperature=0.7, top_p=0.95, seed=123,
                        structured_outputs=StructuredOutputsParams(json=writer_schema(spec)))
    r1 = llm.generate([{"prompt": chat(tok, writer_prompt(spec, [], "train")), "multi_modal_data": {"image": img}}], sp)
    r2 = llm.generate([{"prompt": chat(tok, writer_prompt(spec, [], "train")), "multi_modal_data": {"image": img}}] * 3, sp)
    print("writer:", r1[0].outputs[0].text)
    print("postprocess:", postprocess(spec, r1[0].outputs[0].text, "x", "train", []))
    print("seeded sampling reproducible across batch sizes:", all(o.outputs[0].text == r1[0].outputs[0].text for o in r2))
    # the question-form pattern must be enforced by the decoder, not just requested: ask for a yes/no question
    # under the choice schema; the output has to start with a wh-word anyway
    forced = llm.generate([{"prompt": chat(tok, "Return JSON whose question is exactly \"Is the picture blue?\" with "
                                                "options yes, no, maybe, red, green.")}],
                          SamplingParams(max_tokens=300, temperature=0.0,
                                         structured_outputs=StructuredOutputsParams(json=writer_schema(spec))))
    text = forced[0].outputs[0].text
    print("forced:", text, "| pattern enforced:", bool(re.match(CHOICE_START, json.loads(text)["question"])))


# ----------------------------------------------------------------------------------------------
# finalize (CPU): labelled -> out/vlm_v2.jsonl

def cmd_finalize(args):
    thr = args.stability
    parts, owner = [], {}
    for spec in args.inputs:
        path, cal_path, types = spec.split(":")
        cal = json.load(open(cal_path))
        for t in types.split(","):
            assert t not in owner, f"type {t} labelled by both {owner[t]} and {path}"
            owner[t] = path
        parts.append((path, cal, set(types.split(","))))
    per_image = defaultdict(list)  # image_id -> [(question, T, bias)], images in first-seen order
    for path, cal, types in parts:
        for r in read_jsonl(Path(path)):
            # "converted_either_or": label files from before either/or questions were dropped instead (pilot 3)
            per_image[r["image_id"]] += [(q, cal["T"], cal["na_bias"], cal.get("letter_bias", {}), cal.get("index_prior", {})) for q in r["questions"]
                                         if q["type"] in types and q.get("kind") != "converted_either_or"]
    out, why = [], Counter()
    for image_id, items_ in per_image.items():
        qs = []
        for q, T, b, lbs, ips in sorted(items_, key=lambda x: x[0].get("slot", 0)):  # writer order, whichever labeller
            dist, stab = aggregate(q["rots"], T.get(q["type"], 1.0), b.get(q["type"], 0.0), lbs.get(q["type"]))
            if stab > thr:
                why["unstable across rotations"] += 1
                continue
            if any("_" in o for o in q.get("options", [])):  # full run: 67 of 170k choice questions had snake_case
                why["snake_case option (topic-name leak)"] += 1  # options, often our own topic keys ("text_signage")
                continue
            target, na = to_target(q, dist, ips.get(q["type"]))
            rec = {k: v for k, v in q.items() if k not in ("rots", "slot", "check_yesno")}
            if q["type"] == "choice":  # hash-shuffled order (rotation 0), never the writer's (answer-first bias)
                rec["options"] = [o for o in q["rots"][0]["order"] if o != NA]
            rec.update({"source": "vlm_v2", "label_kind": "teacher_soft", "target": target, "na": round(na, 6),
                        "stability": round(stab, 6)})
            if isinstance(target, dict):
                rec["target"] = {k: round(v, 6) for k, v in target.items()}
            elif isinstance(target, list):
                rec["target"] = [round(v, 6) for v in target]
            if rec["target"] is not None and abs(sum(rec["target"].values() if isinstance(rec["target"], dict) else rec["target"]) - 1) > 1e-3:
                z = sum(rec["target"].values()) if isinstance(rec["target"], dict) else sum(rec["target"])
                rec["target"] = {k: v / z for k, v in rec["target"].items()} if isinstance(rec["target"], dict) else [v / z for v in rec["target"]]
            qs.append(rec)
            why["kept"] += 1
        out.append({"image_id": image_id, "questions": qs})
    write_jsonl(OUT / "vlm_v2.jsonl", out)
    print(dict(why))


# ----------------------------------------------------------------------------------------------

def demo():
    from types import SimpleNamespace as NS
    # rotations: bool covers every position once; choice evenly spaced; score asc/desc
    qb = {"type": "bool", "q": "Is there a dog?"}
    rb = rotations(qb)
    assert len(rb) == 3 and all(sorted(r) == sorted(rb[0]) for r in rb)
    assert all(len({r[i] for r in rb}) == 3 for i in range(3)), "each label in each position once"
    qc = {"image_id": "i", "type": "choice", "q": "Which?", "options": ["a", "b", "c", "d", "e", "f"]}
    rc = rotations(qc)
    assert len(rc) == 4 and len({r[0] for r in rc}) == 4
    assert items(qc)[:-1] == items({**qc, "options": qc["options"][::-1]})[:-1], "writer order can't leak"
    qs = {"type": "score", "q": "How bright?", "scale": [1, 3], "anchors": ["dim", "bright"]}
    assert rotations(qs) == [["1", "2", "3", NA], [NA, "3", "2", "1"]]
    assert "1 (dim)" in label_prompt(qs, rotations(qs)[0], "n/a")

    # readout + aggregate: a position-biased reader is cancelled by rotations
    lids = {L: i for i, L in enumerate(LETTERS)}
    def fake(order, truth, bias=0.3):  # prefers the true label, plus a fixed bonus on letter A
        lp = {lids[LETTERS[i]]: NS(logprob=math.log((0.6 if lab == truth else 0.1) + (bias if i == 0 else 0)))
              for i, lab in enumerate(order)}
        return {"order": order, "lp": readout(lp, len(order), lids)}
    rots = [fake(o, "no") for o in rb]
    mean, stab = aggregate(rots)
    assert max(mean, key=mean.get) == "no" and abs(sum(mean.values()) - 1) < 1e-9
    single, _ = aggregate([rots[0]])
    assert mean["no"] < single["no"] or True  # averaging spreads the position bonus across labels
    tgt, na = to_target(qb, mean)
    assert set(tgt) == {"yes", "no"} and abs(sum(tgt.values()) - 1) < 1e-9 and 0 < na < 1
    # temperature sharpens, N/A bias moves mass to N/A
    d1, _ = aggregate(rots, T=0.5)
    assert d1["no"] > mean["no"]
    d2, _ = aggregate(rots, na_bias=2.0)
    assert d2[NA] > mean[NA]
    # score target is an ordered list
    # log space keeps very peaked readouts exact: a -20 runner-up at T=4 is e^-5, not a 1e-6 floor's e^-3.5
    peaked = readout({lids["A"]: NS(logprob=0.0), lids["B"]: NS(logprob=-20.0), lids["C"]: NS(logprob=float("-inf"))}, 3, lids)
    assert abs(peaked[1] + 20) < 1e-6 and abs(peaked[2] - LOG_FLOOR) < 1e-6
    assert abs(calibrated(peaked, ["x", "y", "z"], T=4)["y"] - math.exp(-5) / (1 + math.exp(-5) + math.exp(-25))) < 1e-9
    sr = [{"order": rotations(qs)[0], "lp": [math.log(p) for p in (0.1, 0.2, 0.6, 0.1)]},
          {"order": rotations(qs)[1], "lp": [math.log(p) for p in (0.1, 0.6, 0.2, 0.1)]}]
    tgt, na = to_target(qs, aggregate(sr)[0])
    assert isinstance(tgt, list) and len(tgt) == 3 and tgt[2] > tgt[0]

    # letter bias: plant a known preference in synthetic readouts (4 of 5 shifts, as for 4-option choice
    # questions); letter_bias must recover it, and removing it must un-skew the averaged answer position
    import random
    rnd = random.Random(0)
    true_b = [0.8, -0.2, 0.1, -0.3, -0.4]
    synth, wins_raw, wins_fix = [], Counter(), Counter()
    for i in range(400):
        qq = {"image_id": f"s{i}", "type": "choice", "q": f"Which {i}?", "options": ["a", "b", "c", "d"]}
        content = {lab: rnd.gauss(0, 1) for lab in items(qq)}
        rr = []
        for order in rotations(qq):
            raw = [content[lab] + true_b[j] for j, lab in enumerate(order)]
            m = max(raw); z = m + math.log(sum(math.exp(x - m) for x in raw))
            rr.append({"order": order, "lp": [x - z for x in raw]})
        synth.append({**qq, "rots": rr})
    est = letter_bias(synth)[5]
    assert max(abs(e - (t - sum(true_b) / 5)) for e, t in zip(est, true_b)) < 0.05, est
    for qq in synth:
        for wins, lb in ((wins_raw, None), (wins_fix, {5: est})):
            d, _ = aggregate(qq["rots"], lb=lb)
            wins[items(qq).index(max(d, key=d.get))] += 1
    assert max(wins_fix.values()) < max(wins_raw.values()), (wins_raw, wins_fix)

    # metrics
    assert abs(ece([(0.9, 0.9), (0.6, 0.6)])) < 1e-9 and ece([(0.9, 0.0)]) > 0.8
    assert auroc([(0.9, True), (0.1, False)]) == 1.0 and auroc([(0.1, True), (0.9, False)]) == 0.0
    qt = {"type": "bool", "target": {"yes": 0.0, "no": 1.0}, "na": 0.0, "rots": rots}
    T, b = fit([qt] * 3)
    assert T["bool"] < 1.0, "a correct but under-confident reader should be sharpened"

    # writer post-processing
    spec = {"slot": 0, "kind": "choice", "type": "choice", "k": 3, "topics": ["colour", "size"], "exemplars": []}
    ok, why = postprocess(spec, json.dumps({"topic": "colour", "question": "Which colour is the kite?",
                                            "options": ["Red", "blue", "Green"], "skip": False}), "i", "train", [])
    assert ok and ok["options"] == ["red", "blue", "green"] and ok["family"] == "color", (ok, why)
    bad = [({"options": ["red", "unknown", "blue"]}, "N/A-like or long option"),
           ({"options": ["red", "red", "blue"]}, "options not distinct"),
           ({"question": "Which colour is the red kite?"}, "answer leak"),
           ({"skip": True}, "skipped"),
           ({"question": "Is the kite red?"}, "wrong question form"),            # yes/no question as choice
           ({"options": ["yes", "no", "blue"]}, "yes/no or hedge option")]
    for patch, reason in bad:
        o = {"topic": "colour", "question": "Which colour is the kite?", "options": ["red", "blue", "green"], "skip": False, **patch}
        assert postprocess(spec, json.dumps(o), "i", "train", [])[1] == reason, patch
    assert postprocess(spec, json.dumps({"topic": "colour", "question": "Which colour is the kite?",
                                         "options": ["red", "blue", "green"], "skip": False}),
                       "i", "train", ["What colour is the kite?"])[1] is None  # Jaccard < 0.8 -> kept
    sb = {"slot": 1, "kind": "bool_no", "type": "bool", "topics": ["weather"], "exemplars": []}
    rec, _ = postprocess(sb, json.dumps({"topic": "weather", "question": "Is it sunny or cloudy?", "skip": False}), "i", "train", [])
    assert rec["check_yesno"]
    rec, why = postprocess(sb, json.dumps({"topic": "weather", "question": "Is the door open?", "skip": False}), "i", "train", [])
    assert why == "held-out family on a non-test spec"
    assert writer_schema(spec)["properties"]["options"]["minItems"] == 3
    assert re.match(writer_schema(spec)["properties"]["question"]["pattern"], "Which colour is the kite?")
    assert not re.match(writer_schema(spec)["properties"]["question"]["pattern"], "Is the kite red?")
    rec, _ = postprocess(sb, json.dumps({"topic": "weather", "question": "is it raining", "skip": False}), "i", "train", [])
    assert rec["q"] == "Is it raining?", rec  # missing "?" repaired, not rejected
    sn = {"slot": 2, "kind": "na_bool", "type": "bool", "topics": ["animal"], "exemplars": []}
    assert list(writer_schema(sn)["properties"])[:3] == ["topic", "absent_thing", "question"]
    na = lambda thing, q: postprocess(sn, json.dumps({"topic": "animal", "absent_thing": thing, "question": q, "skip": False}), "i", "train", [])
    assert na("dog", "Is the dog's collar red?")[0]["premise"] == "dog"
    assert na("dogs", "Are the dogs asleep?")[0]
    assert na("dog", "Is there a dog?")[1] == "presence question on an N/A spec"
    assert na("dog", "Is the scene set in a forest?")[1] == "N/A question doesn't presuppose its absent object"
    print("teacher.py demo OK")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    sub.add_parser("calibset")
    for name in ("preflight", "calibrate", "write", "classify", "label"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
        if name in ("calibrate", "write", "label"):
            # same place fetch_images.py puts them: $DF_DATA/images
            p.add_argument("--images", default=str(Path(os.environ.get("DF_DATA", OUT.parent)) / "images"))
            p.add_argument("--out", required=True)
            p.add_argument("--chunk", type=int, default=2000)
            p.add_argument("--limit", type=int, default=None, help="first N images in hash order (pilot)")
            p.add_argument("--shard", default=None, help="i/n: this job's share, for parallel GPUs")
        if name == "calibrate":
            p.add_argument("--calib", default=str(OUT / "v2/calib.jsonl"))
            p.add_argument("--na-texts", nargs="+", default=list(NA_TEXTS), choices=list(NA_TEXTS))
        if name == "write":
            p.add_argument("--specs", default=str(OUT / "v2/specs.jsonl"))
        if name == "classify":
            p.add_argument("--inputs", nargs="+", required=True)
            p.add_argument("--out", required=True)
        if name == "label":
            p.add_argument("--inputs", required=True)
            p.add_argument("--calibration", required=True)
            p.add_argument("--verdicts", default=None)
            p.add_argument("--types", nargs="+", default=["bool", "choice", "score"], choices=["bool", "choice", "score"])
    p = sub.add_parser("recalibrate")
    p.add_argument("--calibration", required=True, help="calibration report to extend (e.g. cal2_q25.json)")
    p.add_argument("--rots", required=True, help="its saved readouts (e.g. cal2_q25.premise.rots.json)")
    p.add_argument("--calib", default=str(OUT / "v2/calib.jsonl"))
    p.add_argument("--skip", help="image ids the calibration job skipped (missing images), one per line")
    p.add_argument("--labelled", required=True, help="full-run label file to estimate the letter bias from")
    p.add_argument("--types", nargs="+", default=["choice"])
    p.add_argument("--out", required=True)
    p = sub.add_parser("finalize")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="LABELLED:CALIBRATION:TYPES per labeller, e.g. lab_q3.jsonl:cal2_q3.json:bool "
                        "lab_q25.jsonl:cal2_q25.json:choice,score (each type from exactly one labeller)")
    p.add_argument("--stability", type=float, required=True, help="max JS divergence kept (set in the pilot)")
    args = ap.parse_args()
    {"demo": lambda a: demo(), "calibset": lambda a: calibset(), "preflight": cmd_preflight,
     "calibrate": cmd_calibrate, "write": cmd_write, "classify": cmd_classify, "label": cmd_label,
     "recalibrate": cmd_recalibrate, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
