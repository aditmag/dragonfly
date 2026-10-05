"""Cache the frozen vision tower's outputs per image: run ViT + merger once, reuse every
epoch. Qwen3-VL also needs the DeepStack features (3 extra feature maps added inside the LLM's first
layers), so each entry is {grid (3,), embeds (N, H), deepstack (3, N, H)}, ~3 MB/image in bf16 for
the 2B model (N ~190 tokens at the 448² px cap).

    uv run python model/cache.py --ids data/ids_dev.txt --images data/images --out data/cache
"""
import argparse
from pathlib import Path

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

SIZE = {"longest_edge": 448 * 448, "shortest_edge": 64 * 64}


@torch.no_grad()
def encode(model, image_processor, images):
    """Vision outputs for a list of PIL images, one dict per image."""
    vis = image_processor(images=images, size=SIZE, return_tensors="pt")
    return encode_pixels(model, vis["pixel_values"], vis["image_grid_thw"])


@torch.no_grad()
def encode_pixels(model, pixel_values, grid_thw, keep_on_device=False):
    """Same, from already-processed pixels (training: the processor runs in data-loader workers and the
    frozen ViT runs here on the GPU, instead of a disk cache)."""
    dev = model.device
    out = model.model.get_image_features(pixel_values.to(dev, model.dtype), grid_thw.to(dev))
    move = (lambda t: t) if keep_on_device else (lambda t: t.cpu())
    return [{"grid": g.cpu(), "embeds": move(e), "deepstack": move(torch.stack([d[i] for d in out.deepstack_features]))}
            for i, (g, e) in enumerate(zip(grid_thw, out.pooler_output))]


def mm_inputs(entry, device, dtype):
    """Model kwargs replacing pixel_values: forward(..., **mm_inputs(entry))."""
    return {"mm_encoder_outputs": {"image": BaseModelOutputWithDeepstackFeatures(
        pooler_output=(entry["embeds"].to(device, dtype),),
        deepstack_features=[(d.to(device, dtype),) for d in entry["deepstack"]],
    )}, "image_grid_thw": entry["grid"][None].to(device)}


def main():
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="one image_id per line")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    improc = AutoProcessor.from_pretrained(args.model).image_processor
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = [l.strip() for l in open(args.ids) if l.strip()]
    todo = [i for i in ids if not (out_dir / f"{i}.pt").exists()]
    print(f"{len(ids) - len(todo)} cached already, {len(todo)} to go", flush=True)
    for b in range(0, len(todo), args.batch):
        batch = todo[b:b + args.batch]
        images = [Image.open(Path(args.images) / f"{i}.jpg").convert("RGB") for i in batch]
        for i, entry in zip(batch, encode(model, improc, images)):
            torch.save({k: v.to(dtype) if v.is_floating_point() else v for k, v in entry.items()},
                       out_dir / f"{i}.pt")
        if (b // args.batch) % 20 == 0:
            print(f"{b + len(batch)}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        # Self-check: hidden states from cached vision outputs == from pixels (float32, so exact
        # up to float noise; real caches are bf16).
        import json

        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        from heads import hidden_states
        from pack import pack

        from pack import check_setup
        MODEL, root, img_dir, dev = check_setup()
        proc = AutoProcessor.from_pretrained(MODEL)
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
        recs = {}
        for line in open(root / "out" / "dataset_selected.jsonl"):
            r = json.loads(line)
            if (img_dir / f"{r['image_id']}.jpg").exists():
                recs[r["image_id"]] = r
            if len(recs) == 3:
                break
        local = sorted(recs)
        images = [Image.open(img_dir / f"{i}.jpg").convert("RGB") for i in local]
        entries = encode(model, proc.image_processor, images)  # batched: also checks the per-image split
        worst = 0.0
        for i, img, e in zip(local, images, entries):
            p = pack(recs[i]["questions"], e["grid"], proc.tokenizer, model.config.image_token_id)
            vis = proc.image_processor(images=[img], size=SIZE, return_tensors="pt").to(dev)
            with torch.no_grad():
                h_pix = hidden_states(model, p, torch.float32, **vis)
                h_cache = hidden_states(model, p, torch.float32, **mm_inputs(e, dev, torch.float32))
            worst = max(worst, (h_pix - h_cache).abs().max().item())
            print(f"  {i}: {e['embeds'].shape[0]} image tokens, deepstack {tuple(e['deepstack'].shape)}")
        print(f"cached vs pixels: worst |Δ hidden| = {worst:.2e}")
        assert worst < 1e-2, worst
        print("cache.py self-check OK")
