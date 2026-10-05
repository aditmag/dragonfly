"""Run the student on ImajevBench v2.0-lite (hf.co/datasets/mohit67890/imajev-bench, CC BY 4.0) and write run
folders the benchmark's own scorer accepts (`python -m imajev_bench score`, Apache-2.0 code from
github.com/mohit67890/imajev on PYTHONPATH).

Interface: our own single-pass heads, like the benchmark's section A2 (other Jev-class models through their own
interface), not its letter-code rotation harness. The model is order-invariant by construction, so one pass equals
"all rotations". Mapping:
  state + question -> one question text ("State: <json>\\nQuestion: ...", the benchmark-neutral wording);
  boolean -> bool; choice -> choice ("value: description"); ordinal -> choice over its levels;
  N/A head -> "__unknown__"; probabilities = (1 - na) * answer distribution; decision = argmax incl. unknown.
Text-only items (no image) get a blank white 64x64 image: the student always reads an image.

    python3 imajevbench.py --bench ../../imajev-bench --run ../runs/full4b --out ../runs/full4b/imajevbench
    (no --run = the base model: step-0 heads equal its Yes/No and option log-likelihood readout)
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from cache import encode
from heads import Heads
from train import forward
from imajev_bench.cli import read_jsonl
from imajev_bench.runner import canonical_bytes, digest, file_digest, run_hashes
from imajev_bench.schema import model_payload, validate_records


def domain(field):
    """(wire key, typed value, option text) per candidate, in the benchmark's order."""
    if field["type"] == "boolean":
        return [("true", True, None), ("false", False, None)]
    items = field["levels"] if field["type"] == "ordinal" else field["options"]
    return [(str(x["value"]), x["value"], f"{x['value']}: {x['description']}" if x.get("description") else str(x["value"]))
            for x in items]


def question(payload):
    req = payload["request"]
    field = req["fields"][0]
    text = f"State: {json.dumps(req['state'], sort_keys=True, ensure_ascii=False)}\nQuestion: {field['question']}"
    if field["type"] == "boolean":
        return {"type": "bool", "q": text}
    return {"type": "choice", "q": text, "options": [t for _, _, t in domain(field)]}


def predict(m, payload, root):
    field = payload["request"]["fields"][0]
    img = [Image.open(root / i["path"]).convert("RGB") for i in payload["images"]][:1] or [Image.new("RGB", (64, 64), "white")]
    t0 = time.perf_counter()
    with torch.no_grad():
        entry = encode(m["model"], m["proc"].image_processor, img)[0]
        q = question(payload)
        o = forward(m["model"], m["heads"], m["tok"], m["model"].config.image_token_id, [q], entry, "cuda", m["dtype"])[0]
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    na = torch.sigmoid(o["na"].float()).item()
    if q["type"] == "bool":
        p = torch.sigmoid(o["bool"].float() / m["temps"].get("bool", 1.0)).item()
        dist = [p, 1 - p]
    else:
        dist = (o["opt"].float() / m["temps"].get("choice", 1.0)).softmax(0).tolist()
    cands = domain(field)
    probs = {k: (1 - na) * d for (k, _, _), d in zip(cands, dist)}
    probs["__unknown__"] = na
    best = max(probs, key=probs.get)
    value = None if best == "__unknown__" else next(v for k, v, _ in cands if k == best)
    return {"status": "abstained" if value is None else "answered", "value": value, "probabilities": probs,
            "confidence_source": "dragonfly_heads" + ("_temperature" if m["temps"] else ""), "latency_ms": ms}


def load(model_name, run, dtype):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=dtype).to("cuda").eval()
    tok = proc.tokenizer
    heads = Heads(model.config.text_config.hidden_size, tok.convert_tokens_to_ids("Yes"), tok.convert_tokens_to_ids("No")).to("cuda")
    temps = {}
    if run:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, Path(run) / "lora").merge_and_unload().eval()
        heads.load_state_dict(torch.load(Path(run) / "heads.pt", map_location="cuda"))
        temps = {t: v["T"] for t, v in json.load(open(Path(run) / "log.json"))["temperatures"].items()}
    return {"model": model, "proc": proc, "tok": tok, "heads": heads.eval(), "temps": temps, "dtype": dtype}


def run_split(m, records, root, out, describe):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"format_version": "0.1.0", "adapter": "dragonfly-heads", **run_hashes(records), "record_count": len(records),
                "reviewed": all(r["annotation_status"] == "reviewed" for r in records), "concurrency": 1,
                "backend": describe, "calibration": "temperatures fitted on dragonfly's own val split" if m["temps"] else None,
                "decision_rule": "argmax over (1 - P(N/A)) x answer distribution and P(N/A) as unknown; one pass, order-invariant",
                "latency_definition": "image encode + one forward, excludes load and disk reads"}
    (out / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
    with (out / "predictions.jsonl").open("x") as pf, (out / "raw.jsonl").open("x") as rf:
        for r in records:
            payload = model_payload(r)
            row = {**predict(m, payload, root), "id": r["id"]}
            pf.write(json.dumps(row, allow_nan=False) + "\n")
            rf.write(json.dumps({"id": r["id"], "payload_sha256": digest(payload), "question": question(payload)}) + "\n")
    (out / "completion.json").write_bytes(canonical_bytes({
        "status": "complete", "completed_count": len(records), "manifest_sha256": file_digest(out / "manifest.json"),
        "predictions_sha256": file_digest(out / "predictions.jsonl"), "raw_sha256": file_digest(out / "raw.jsonl")}) + b"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="dataset snapshot dir (records/, assets/)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--run", help="training output dir (lora/, heads.pt, log.json); omit for the base model")
    ap.add_argument("--splits", default="dev,calibration,test")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.bench)
    all_records = list(validate_records(read_jsonl(root / "records/records-public.jsonl"), root, require_reviewed=False))
    m = load(args.model, args.run, getattr(torch, args.dtype))
    describe = {"model": args.model, "run": args.run, "dtype": args.dtype, "device": torch.cuda.get_device_name()}
    for split in args.splits.split(","):
        records = [r for r in all_records if r["split"] == split]
        run_split(m, records, root, Path(args.out) / split, describe)
        print(split, len(records), "done", flush=True)


if __name__ == "__main__":
    main()
