"""Evaluation: trained student vs the baseline (step 0 = the base model's
own readout through the same heads, warm-started), on held-out images.

Per question type: accuracy (soft: the target's mass on the predicted answer), ECE (15 bins), NLL; N/A
AUROC; POPE (yes-bias) accuracy and yes-rate; the held-out families (material, open_closed). Both models
get a temperature per type fitted on val images (train.fit_temperatures), and are reported with and
without it.

    python model/evaluate.py --ids data/out/ids_test.txt --val-ids data/out/ids_val.txt --val-limit 1000 \
        --images data/images --adapter runs/full4b --out runs/full4b/eval_test.json
    (no --adapter = the baseline)
"""
import argparse
import bisect
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from heads import Heads
from pack import answers
from train import ROOT, Data, evaluate, fit_temperatures

HELD_OUT = {"material", "open_closed"}


def probs(o, q, T):
    """Predicted answer distribution (list, in answers(q) order) and P(N/A)."""
    na = torch.sigmoid(o["na"]).item()
    if q["type"] == "bool":
        p = torch.sigmoid(o["bool"] / T).item()
        return [p, 1 - p], na
    return (o["opt"] / T).softmax(0).tolist(), na


def target_list(q):
    t = q["target"]
    if q["type"] == "bool":
        z = t["yes"] + t["no"]
        return [t["yes"] / z, t["no"] / z]
    v = t if q["type"] == "score" else [t[a] for a in answers(q)]
    z = sum(v)
    return [x / z for x in v]


def ece(pairs, bins=15):
    by = defaultdict(list)
    for c, a in pairs:
        by[min(int(c * bins), bins - 1)].append((c, a))
    return sum(len(v) / len(pairs) * abs(sum(c for c, _ in v) / len(v) - sum(a for _, a in v) / len(v))
               for v in by.values()) if pairs else None


def auroc(scores_labels):
    pos = sorted(s for s, y in scores_labels if y)
    neg = sorted(s for s, y in scores_labels if not y)
    if not pos or not neg:
        return None
    u = sum(bisect.bisect_left(neg, s) + 0.5 * (bisect.bisect_right(neg, s) - bisect.bisect_left(neg, s)) for s in pos)
    return u / (len(pos) * len(neg))


def metrics(raw, temps):
    """raw: [(outputs, question)]. Groups: type, held-out families, POPE, label kind."""
    groups = defaultdict(list)
    na_pairs = []
    for o, q in raw:
        na_pairs.append((torch.sigmoid(o["na"]).item(), q["na"] >= 0.5))
        if q["target"] is None or q["na"] >= 0.5:
            continue
        T = temps.get(q["type"], {}).get("T", 1.0)
        p, _ = probs(o, q, T)
        t = target_list(q)
        k = max(range(len(p)), key=p.__getitem__)
        nll = -sum(a * math.log(max(b, 1e-12)) for a, b in zip(t, p))
        row = (p[k], t[k], nll, k == 0 if q["type"] == "bool" else None, q)  # bool: k == 0 means "yes"
        groups[q["type"]].append(row)
        groups[f"label_kind/{q['label_kind']}/{q['type']}"].append(row)
        if q["family"] in HELD_OUT:
            groups[f"held_out/{q['family']}"].append(row)
        if q["source"].startswith("pope"):
            groups[q["source"]].append(row)
    out = {}
    for g, rows in sorted(groups.items()):
        m = {"n": len(rows), "acc": sum(r[1] for r in rows) / len(rows), "ece": ece([(r[0], r[1]) for r in rows]),
             "nll": sum(r[2] for r in rows) / len(rows)}
        if g.startswith("pope"):
            m["yes_rate"] = sum(bool(r[3]) for r in rows) / len(rows)
        out[g] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
    out["na_auroc"] = round(auroc(na_pairs), 4) if auroc(na_pairs) is not None else None
    return out


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--ids", required=True, help="images to evaluate on (e.g. test)")
    ap.add_argument("--val-ids", required=True, help="images for the temperature fit")
    ap.add_argument("--images", required=True)
    ap.add_argument("--dataset", default=str(ROOT / "data/out/dataset_selected.jsonl"))
    ap.add_argument("--adapter", help="training output dir (lora/ + heads.pt); omit for the baseline")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8, help="images per forward pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", help="float16 for GPUs without bf16 (e.g. P100)")
    args = ap.parse_args()

    dev, dtype = "cuda", getattr(torch, args.dtype)
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    dataset = {r["image_id"]: r for r in map(json.loads, open(args.dataset))}
    test = Data(args.ids, dataset, None, args.images, proc.image_processor)
    val = Data(args.val_ids, dataset, None, args.images, proc.image_processor)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    heads = Heads(model.config.text_config.hidden_size, tok.convert_tokens_to_ids("Yes"),
                  tok.convert_tokens_to_ids("No")).to(dev)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, Path(args.adapter) / "lora").eval()
        heads.load_state_dict(torch.load(Path(args.adapter) / "heads.pt", map_location=dev))
        base = model.base_model.model  # the Qwen3VL model with LoRA injected
    else:
        base = model
    img_id = base.config.image_token_id

    _, val_raw, _ = evaluate(base, heads, tok, img_id, val, dev, dtype, args.val_limit, args.workers, args.batch)
    temps = fit_temperatures(val_raw)
    loss, raw, per = evaluate(base, heads, tok, img_id, test, dev, dtype, args.limit, args.workers, args.batch)
    report = {"model": args.model, "adapter": args.adapter, "images": min(args.limit or len(test), len(test)),
              "loss": round(loss, 4), "loss_by_type": per, "temperatures": temps,
              "raw_T1": metrics(raw, {}), "with_T": metrics(raw, temps)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("loss", "loss_by_type")}), flush=True)
    for name in ("raw_T1", "with_T"):
        print(name, json.dumps({g: v for g, v in report[name].items() if "/" not in g}), flush=True)


if __name__ == "__main__":
    main()
