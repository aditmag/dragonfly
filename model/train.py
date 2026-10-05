"""Train LoRA + typed heads on packed sequences.

One micro-step = one image with all its questions packed into one sequence; --accum images per
optimizer step. Losses are proper scoring rules against the dataset's (soft) targets:
  N/A     BCE on every question
  bool    BCE vs P(yes)                         } weighted by (1 - target N/A)
  choice  cross-entropy vs the soft distribution }
  score   cross-entropy + ranked probability score (ordinal: CDF distance)
After training, a temperature per head type is fitted on the val images (post-hoc calibration).

Vision: either --cache (precomputed by cache.py, for small dev runs) or --images (data-loader
workers decode + preprocess, the frozen ViT runs on the GPU without gradients each step; a 4B cache of
all 142k images would be ~550 GB, and caching only pays across
epochs). Data order is a seeded permutation per epoch, so --resume continues exactly where a run stopped.

    uv run python model/train.py --model Qwen/Qwen3-VL-4B-Instruct --train-ids data/out/ids_train.txt \
        --val-ids data/out/ids_val.txt --images data/images --out runs/full4b --epochs 1 --accum 16 --batch 8 \
        --warmup 200 --final-val-limit 2000 --workers 16
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cache import SIZE, encode_pixels, mm_inputs
from heads import Heads, hidden_states, hidden_states_batch
from pack import answers, pack

ROOT = Path(__file__).parent.parent


def question_loss(o, q):
    loss = F.binary_cross_entropy_with_logits(o["na"], torch.tensor(float(q["na"]), device=o["na"].device))
    t = q["target"]
    if t is None:
        return loss
    w = 1 - q["na"]
    dev = o["na"].device
    if q["type"] == "bool":
        p_yes = t["yes"] / (t["yes"] + t["no"])
        return loss + w * F.binary_cross_entropy_with_logits(o["bool"], torch.tensor(p_yes, device=dev))
    tgt = torch.tensor(t if q["type"] == "score" else [t[a] for a in answers(q)], device=dev, dtype=torch.float32)
    tgt = tgt / tgt.sum()
    logp = o["opt"].log_softmax(0)
    loss = loss + w * -(tgt * logp).sum()
    if q["type"] == "score":
        rps = ((logp.exp().cumsum(0) - tgt.cumsum(0)) ** 2).sum() / max(len(tgt) - 1, 1)
        loss = loss + w * rps
    return loss


class Data(torch.utils.data.Dataset):
    """Image ids with a record and an image (or cache entry). Items: (image_id, vision), where vision is
    a cache entry or {"pixel_values", "grid"} preprocessed in the worker."""

    def __init__(self, ids_file, dataset, cache_dir=None, images_dir=None, improc=None):
        ids = [l.strip() for l in open(ids_file) if l.strip()]
        src = Path(cache_dir or images_dir)
        ext = ".pt" if cache_dir else ".jpg"
        self.ids = [i for i in ids if i in dataset and (src / f"{i}{ext}").exists()]
        self.cache_dir, self.images_dir, self.improc, self.dataset = cache_dir, images_dir, improc, dataset
        if len(self.ids) < len(ids):
            print(f"{ids_file}: {len(ids) - len(self.ids)} ids skipped (no image/cache entry or record)")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image_id = self.ids[i]
        if self.cache_dir:
            return image_id, torch.load(Path(self.cache_dir) / f"{image_id}.pt")
        from PIL import Image
        vis = self.improc(images=[Image.open(Path(self.images_dir) / f"{image_id}.jpg").convert("RGB")],
                          size=SIZE, return_tensors="pt")
        return image_id, {"pixel_values": vis["pixel_values"], "grid": vis["image_grid_thw"][0]}

    def loader(self, order, workers):
        """Items in the given index order, prefetched by worker processes."""
        return torch.utils.data.DataLoader(self, sampler=order, batch_size=None, num_workers=workers,
                                           prefetch_factor=8 if workers else None, persistent_workers=False)


def vision_inputs(model, vision, dev, dtype):
    if "pixel_values" in vision:  # frozen ViT on the fly, no gradients
        vision = encode_pixels(model, vision["pixel_values"], vision["grid"][None], keep_on_device=True)[0]
    return vision, mm_inputs(vision, dev, dtype)


def forward(model, heads, tok, img_id, questions, vision, dev, dtype):
    entry, mm = vision_inputs(model, vision, dev, dtype)
    p = pack(questions, entry["grid"], tok, img_id)
    h = hidden_states(model, p, dtype, **mm)
    return heads(h, p, model.lm_head)


def forward_batch(model, heads, tok, img_id, items, dataset, dev, dtype):
    """Several images in one padded forward (heads.hidden_states_batch); items: [(image_id, vision)].
    Returns [(questions, outputs)] per image. The frozen ViT also runs once for all of them."""
    if "pixel_values" in items[0][1]:
        entries = encode_pixels(model, torch.cat([v["pixel_values"] for _, v in items]),
                                torch.stack([v["grid"] for _, v in items]), keep_on_device=True)
    else:
        entries = [v for _, v in items]
    qss = [dataset[i]["questions"] for i, _ in items]
    packs = [pack(qs, e["grid"], tok, img_id) for qs, e in zip(qss, entries)]
    hs = hidden_states_batch(model, packs, entries, dtype, tok.pad_token_id)
    return [(qs, heads(h, p, model.lm_head)) for qs, h, p in zip(qss, hs, packs)]


def split_by_tokens(items, dataset, tok, img_id, budget):
    """Split a group of images into forward passes whose padded size (images x longest sequence) stays
    within `budget` tokens. Full run: a group of 8 long images (~16k padded tokens) ran the GPU out of
    memory at step ~5,600, while a typical group is ~5k. Sorting by length first keeps padding low."""
    lens = [len(pack(dataset[i]["questions"], v["grid"], tok, img_id).input_ids) for i, v in items]
    order = sorted(range(len(items)), key=lens.__getitem__)
    out, cur, longest = [], [], 0
    for k in order:
        if cur and max(longest, lens[k]) * (len(cur) + 1) > budget:
            out.append(cur)
            cur, longest = [], 0
        cur.append(items[k])
        longest = max(longest, lens[k])
    return out + [cur]


def batches(it, n):
    """Group an iterator of items into lists of n."""
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


@torch.no_grad()
def evaluate(model, heads, tok, img_id, data, dev, dtype, limit=None, workers=0, batch=1):
    """Mean question loss on the first `limit` val images, overall and per type; raw outputs for the
    temperature fit."""
    model.eval()
    losses, raw = [], []
    for items in batches(data.loader(list(range(min(limit or len(data), len(data)))), workers), batch):
        for qs, outs in forward_batch(model, heads, tok, img_id, items, data.dataset, dev, dtype):
            for o, q in zip(outs, qs):
                losses.append((q["type"], question_loss(o, q).item()))
                raw.append(({k: v.detach().cpu() for k, v in o.items()}, q))
    model.train()
    by = {t: [l for tt, l in losses if tt == t] for t in ("bool", "choice", "score")}
    per_type = {t: round(sum(v) / len(v), 4) for t, v in by.items() if v}
    return sum(l for _, l in losses) / len(losses), raw, per_type


def fit_temperatures(raw):
    """Post-hoc temperature per head type on held-out outputs (grid search, NLL of the answer)."""
    grid = [math.exp(x / 20) for x in range(-40, 41)]  # T in [0.14, 7.4]
    temps = {}
    for typ in ("bool", "choice", "score"):
        items = [(o, q) for o, q in raw if q["type"] == typ and q["target"] is not None]
        if not items:
            continue

        def nll(T):
            total = 0.0
            for o, q in items:
                if typ == "bool":
                    p = q["target"]["yes"] / (q["target"]["yes"] + q["target"]["no"])
                    total += F.binary_cross_entropy_with_logits(o["bool"] / T, torch.tensor(p)).item()
                else:
                    t = torch.tensor(q["target"] if typ == "score" else [q["target"][a] for a in answers(q)])
                    total += -(t / t.sum() * (o["opt"] / T).log_softmax(0)).sum().item()
            return total / len(items)

        best = min(grid, key=nll)
        temps[typ] = {"T": best, "nll_T1": nll(1.0), "nll_best": nll(best), "n": len(items)}
    return temps


def order_for(n_items, start_step, steps, accum, seed):
    """Item indices for optimizer steps start_step+1 .. steps: epoch e is a seeded permutation of all
    items, so the order is fixed by (seed, step) and a resumed run sees exactly the same images."""
    perms, out = {}, []
    for idx in range(start_step * accum, steps * accum):
        e = idx // n_items
        if e not in perms:
            perms[e] = random.Random(f"{seed}-{e}").sample(range(n_items), n_items)
        out.append(perms[e][idx % n_items])
    return out


def main():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--val-ids", required=True)
    ap.add_argument("--cache", help="cached vision features (cache.py); or --images")
    ap.add_argument("--images", help="image dir: run the frozen ViT on the fly")
    ap.add_argument("--workers", type=int, default=8, help="data-loader workers (with --images)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=str(ROOT / "data/out/dataset_selected.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--dtype", default=None, help="default: bfloat16 on cuda, float32 elsewhere")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--epochs", type=float, default=None, help="overrides --steps: epochs over the train images")
    ap.add_argument("--accum", type=int, default=8, help="images per optimizer step")
    ap.add_argument("--batch", type=int, default=1, help="images per forward pass (divides --accum)")
    ap.add_argument("--max-tokens", type=int, default=7000, help="padded tokens per forward pass (training)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100, help="linear warmup steps, then cosine to 10%%")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-limit", type=int, default=100)
    ap.add_argument("--final-val-limit", type=int, default=None, help="val images for the final eval + temperatures")
    ap.add_argument("--save-every", type=int, default=500, help="checkpoint (resumable) every N steps")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/ckpt.pt if present")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert bool(args.cache) != bool(args.images), "give exactly one of --cache / --images"
    assert args.accum % args.batch == 0, "--batch must divide --accum"

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.bfloat16 if dev == "cuda" else torch.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    dataset = {r["image_id"]: r for r in map(json.loads, open(args.dataset))}
    train = Data(args.train_ids, dataset, args.cache, args.images, proc.image_processor)
    val = Data(args.val_ids, dataset, args.cache, args.images, proc.image_processor)
    workers = args.workers if args.images else 0
    steps = math.ceil(args.epochs * len(train) / args.accum) if args.epochs else args.steps
    print(f"train {len(train)} images, val {len(val)} images, {steps} steps x {args.accum} images, "
          f"device {dev}, dtype {dtype}", flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=dtype).to(dev)
    model.requires_grad_(False)
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules=r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
    peft_model = get_peft_model(model, lora)  # injects LoRA into `model` in place; forward uses `model`
    heads = Heads(model.config.text_config.hidden_size, tok.convert_tokens_to_ids("Yes"),
                  tok.convert_tokens_to_ids("No")).to(dev)
    img_id = model.config.image_token_id
    lora_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable: LoRA {sum(p.numel() for p in lora_params) / 1e6:.1f}M "
          f"({lora_params[0].dtype}), heads {sum(p.numel() for p in heads.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr},
                             {"params": heads.parameters(), "lr": args.head_lr}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) if s < args.warmup else
                                              0.1 + 0.45 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, steps - args.warmup))))

    log, start = [], 0
    ckpt = out / "ckpt.pt"
    if args.resume and ckpt.exists():
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        set_peft_model_state_dict(peft_model, ck["lora"])
        heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, log = ck["step"], ck["log"]
        print(f"resumed from step {start}", flush=True)
    else:
        val_loss, _, per = evaluate(model, heads, tok, img_id, val, dev, dtype, args.val_limit, workers, args.batch)
        print(f"step 0: val loss {val_loss:.4f} {per} (= baseline + N/A prior)", flush=True)
        log.append({"step": 0, "val": val_loss, "val_by_type": per})

    run, t0 = [], time.time()
    model.train()
    items = iter(train.loader(order_for(len(train), start, steps, args.accum, args.seed), workers))
    for step in range(start + 1, steps + 1):
        for _ in range(args.accum // args.batch):  # same loss as one image at a time: mean over images
            group = [next(items) for _ in range(args.batch)]
            for sub in split_by_tokens(group, dataset, tok, img_id, args.max_tokens):  # backward per pass
                per_image = [torch.stack([question_loss(o, q) for o, q in zip(outs, qs)]).mean()
                             for qs, outs in forward_batch(model, heads, tok, img_id, sub, dataset, dev, dtype)]
                (torch.stack(per_image).sum() / args.accum).backward()
                run += [l.item() for l in per_image]
        torch.nn.utils.clip_grad_norm_(lora_params + list(heads.parameters()), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 10 == 0:
            rate = (time.time() - t0) / (step - start)
            mem = f"  peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if dev == "cuda" else ""
            print(f"step {step}/{steps}: train loss {sum(run[-80:]) / len(run[-80:]):.4f}  {rate:.2f}s/step  "
                  f"lr {sched.get_last_lr()[0]:.2e}  ETA {(steps - step) * rate / 3600:.1f} h{mem}", flush=True)
        if step % args.eval_every == 0 and step < steps:
            val_loss, _, per = evaluate(model, heads, tok, img_id, val, dev, dtype, args.val_limit, workers, args.batch)
            print(f"step {step}: val loss {val_loss:.4f} {per}", flush=True)
            log.append({"step": step, "val": val_loss, "val_by_type": per, "train": sum(run[-80:]) / len(run[-80:])})
        if step % args.save_every == 0 and step < steps:
            torch.save({"lora": get_peft_model_state_dict(peft_model), "heads": heads.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "log": log}, ckpt)
            print(f"step {step}: checkpoint saved", flush=True)

    val_loss, raw, per = evaluate(model, heads, tok, img_id, val, dev, dtype, args.final_val_limit or args.val_limit, workers, args.batch)
    print(f"final: val loss {val_loss:.4f} {per}", flush=True)
    log.append({"step": steps, "val": val_loss, "val_by_type": per, "train": sum(run[-80:]) / max(1, len(run[-80:]))})
    temps = fit_temperatures(raw)
    print("temperatures:", json.dumps(temps, indent=1))
    peft_model.save_pretrained(out / "lora")  # adapter weights only, not the base model
    torch.save(heads.state_dict(), out / "heads.pt")
    json.dump({"args": vars(args), "log": log, "temperatures": temps}, open(out / "log.json", "w"), indent=1)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
