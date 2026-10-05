"""Pack one image + all its typed questions into a single sequence.

    [prefix: <|im_start|>user\n<|vision_start|> image tokens <|vision_end|>]
    [Q1 block][Q2 block]...[option blocks of choice/score questions]

- Q block = "{prompt}<|im_end|>\n<|im_start|>assistant\n" (the model's own chat template). Its last
  token is the DECIDE readout: a bool question reads its Yes/No logits there.
- Option block = option tokens + <|im_end|>, read as the answer continuation of its question. Its
  log-likelihood is the option's baseline score; its last token is the OPT readout.
- Isolation mask: prefix is causal; a Q block sees prefix + itself; an option block sees prefix +
  its own question + itself. Every Q block starts at the same position P0, every option right
  after its question, and choice prompts list options sorted -- so outputs can't depend on question
  or option order, and each readout equals running that question alone (checked in __main__).

Image-agnostic: needs only the image's grid (t, h, w), so it works with cached vision features.
"""
from dataclasses import dataclass, field

import torch

PREFIX = "<|im_start|>user\n<|vision_start|>"
Q_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
END = "<|im_end|>"
MERGE = 2  # Qwen3-VL spatial merge size


def prompt_for(q):
    if q["type"] == "bool":
        return f"{q['q']}\nAnswer Yes or No."
    if q["type"] == "choice":  # sorted, so the prompt depends on the option set, not its order
        return f"{q['q']}\nOptions: {', '.join(sorted(q['options']))}.\nAnswer with one of the options."
    lo, hi = q["scale"]
    a, b = q.get("anchors") or (str(lo), str(hi))
    return f"{q['q']}\nAnswer with a number from {lo} ({a}) to {hi} ({b})."


def answers(q):
    """Answer strings in the record's own order (targets index into this)."""
    if q["type"] == "bool":
        return ["Yes", "No"]
    if q["type"] == "choice":
        return list(q["options"])
    lo, hi = q["scale"]
    return [str(v) for v in range(lo, hi + 1)]


@dataclass
class Packed:
    input_ids: torch.Tensor      # (L,)
    position_ids: torch.Tensor   # (3, L) M-RoPE t/h/w
    allowed: torch.Tensor        # (L, L) bool, True = may attend
    decide: list = field(default_factory=list)  # per question: index of its DECIDE token
    options: list = field(default_factory=list)  # per question: [(start, end, token_ids)] or []

    def mask(self, dtype):
        """Additive 4D mask (1, 1, L, L) for the model (0 = attend, min = blocked)."""
        m = torch.zeros(self.allowed.shape, dtype=dtype).masked_fill(~self.allowed, torch.finfo(dtype).min)
        return m[None, None]


def prefix_positions(n_text, grid):
    """Positions for PREFIX text, then the image grid, then <|vision_end|>; same rule as
    Qwen3VLModel.get_rope_index (text: running counter; image: t/h/w offset from the counter)."""
    t, h, w = (int(x) for x in grid)
    h, w = h // MERGE, w // MERGE
    text = torch.arange(n_text).expand(3, -1)
    tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
    img = torch.stack([tt, hh, ww]).reshape(3, -1) + n_text
    end = n_text + max(h, w)
    return torch.cat([text, img, torch.full((3, 1), end)], dim=1), end + 1


def pack(questions, grid, tok, image_token_id):
    pre_ids = tok.encode(PREFIX, add_special_tokens=False)
    n_img = int(grid[0] * grid[1] * grid[2]) // MERGE**2
    ids = pre_ids + [image_token_id] * n_img + tok.convert_tokens_to_ids(["<|vision_end|>"])
    pos, p0 = prefix_positions(len(pre_ids), grid)
    pos = [pos]
    blk = [-1] * len(ids)  # per token: its block id (-1 = prefix)
    par = [-2] * len(ids)  # per token: its block's parent question block (-2 = none)

    out = Packed(None, None, None)
    q_len = []
    for i, q in enumerate(questions):
        block = tok.encode(prompt_for(q) + Q_SUFFIX, add_special_tokens=False)
        q_len.append(len(block))
        ids += block
        pos.append(torch.arange(p0, p0 + len(block)).expand(3, -1))
        blk += [i] * len(block)
        par += [-2] * len(block)
        out.decide.append(len(ids) - 1)
    end_id = tok.convert_tokens_to_ids(END)
    next_blk = len(questions)
    for i, q in enumerate(questions):
        opts = []
        if q["type"] != "bool":
            start_pos = p0 + q_len[i]
            for a in answers(q):
                toks = tok.encode(a, add_special_tokens=False) + [end_id]
                opts.append((len(ids), len(ids) + len(toks), toks))
                ids += toks
                pos.append(torch.arange(start_pos, start_pos + len(toks)).expand(3, -1))
                blk += [next_blk] * len(toks)
                par += [i] * len(toks)
                next_blk += 1
        out.options.append(opts)

    L = len(ids)
    b, pa = torch.tensor(blk), torch.tensor(par)
    see = (b[None, :] == -1) | (b[None, :] == b[:, None]) | (b[None, :] == pa[:, None])
    out.allowed = torch.ones(L, L, dtype=torch.bool).tril() & see
    out.input_ids = torch.tensor(ids)
    out.position_ids = torch.cat(pos, dim=1)
    return out


def check_setup():
    """Model, data dir, image dir and device for the self-checks (pack / heads / cache __main__):
    DF_MODEL (default the 2B dev model) and DF_IMAGES (default data/images) override, so the same
    checks run on any machine with the 4B model."""
    import os
    from pathlib import Path
    root = Path(__file__).parent.parent / "data"
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return (os.environ.get("DF_MODEL", "Qwen/Qwen3-VL-2B-Instruct"), root,
            Path(os.environ.get("DF_IMAGES", root / "images")), dev)


def check_record(root, img_dir, skip=0):
    """A record with an image on disk whose questions cover bool, choice and score (5 questions)."""
    import json
    for line in open(root / "out" / "dataset_selected.jsonl"):
        r = json.loads(line)
        if {"bool", "choice", "score"} <= {q["type"] for q in r["questions"]} and (img_dir / f"{r['image_id']}.jpg").exists():
            if skip == 0:
                by = {}
                for q in r["questions"]:  # keep all three types inside the 5
                    by.setdefault(q["type"], q)
                rest = [q for q in r["questions"] if q not in by.values()]
                return r["image_id"], (list(by.values()) + rest)[:5]
            skip -= 1
    raise SystemExit("no image on disk with bool+choice+score questions")


if __name__ == "__main__":
    # Self-check against the real model: packed readouts == each
    # question run alone through the stock HF forward; shuffling question/option order changes
    # nothing; our prefix positions == Qwen's own get_rope_index.
    import random

    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    MODEL, root, img_dir, dev = check_setup()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    tok, img_id = proc.tokenizer, model.config.image_token_id
    SIZE = {"longest_edge": 448 * 448, "shortest_edge": 64 * 64}

    image_id, qs = check_record(root, img_dir)
    rec = {"image_id": image_id}
    image = Image.open(img_dir / f"{image_id}.jpg").convert("RGB")
    vis = proc.image_processor(images=[image], size=SIZE, return_tensors="pt").to(dev)
    grid = vis["image_grid_thw"][0]
    print(rec["image_id"], [q["type"] for q in qs], "grid", grid.tolist())

    def run_packed(questions):
        p = pack(questions, grid, tok, img_id)
        with torch.no_grad():
            lp = model(input_ids=p.input_ids[None].to(dev), pixel_values=vis["pixel_values"],
                       image_grid_thw=vis["image_grid_thw"], position_ids=p.position_ids[:, None].to(dev),
                       attention_mask=p.mask(torch.float32).to(dev)).logits[0].float().log_softmax(-1).cpu()
        res = []
        for i, q in enumerate(questions):
            r = {"decide": lp[p.decide[i]]}
            r["opts"] = [sum(lp[(p.decide[i] if k == 0 else s + k - 1)][t].item() for k, t in enumerate(toks))
                         for s, e, toks in p.options[i]]
            res.append(r)
        return p, res

    # 1. positions match Qwen's own rope index
    p, res = run_packed(qs)
    text = "".join(["<|im_start|>user\n<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "x"])
    ref_in = proc(text=[text], images=[image], images_kwargs={"size": SIZE}, return_tensors="pt")
    ref_pos, _ = model.model.get_rope_index(ref_in["input_ids"], mm_token_type_ids=ref_in["mm_token_type_ids"],
                                            image_grid_thw=ref_in["image_grid_thw"])
    n_pre = (p.input_ids == tok.convert_tokens_to_ids("<|vision_end|>")).nonzero()[0].item() + 1
    assert torch.equal(ref_pos[:, 0, :n_pre], p.position_ids[:, :n_pre]), "prefix positions differ from get_rope_index"
    assert torch.equal(ref_in["input_ids"][0, :n_pre], p.input_ids[:n_pre]), "prefix ids differ from processor"
    print("positions + prefix ids match processor/get_rope_index")

    # 2. packed == separate
    worst = 0.0
    for i, q in enumerate(qs):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_for(q)}]}]
        chat = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inp = proc(text=[chat], images=[image], images_kwargs={"size": SIZE}, return_tensors="pt").to(dev)
        with torch.no_grad():
            ref = model(**inp).logits[0, -1].float().log_softmax(-1).cpu()
        d = (ref - res[i]["decide"]).abs().max().item()
        worst = max(worst, d)
        for a, packed_ll in zip(answers(q), res[i]["opts"]):
            inp_o = proc(text=[chat + a + END], images=[image], images_kwargs={"size": SIZE}, return_tensors="pt").to(dev)
            n = inp["input_ids"].shape[1]
            with torch.no_grad():
                lp_o = model(**inp_o).logits[0].float().log_softmax(-1).cpu()
            ref_ll = sum(lp_o[n - 1 + k, t].item() for k, t in enumerate(inp_o["input_ids"][0, n:].tolist()))
            worst = max(worst, abs(ref_ll - packed_ll))
        print(f"  q{i} {q['type']:6s} decide max|Δ|={d:.1e}  {q['q'][:60]!r}")
    print(f"packed vs separate: worst |Δ log-prob| = {worst:.2e}")
    assert worst < 1e-3, worst

    # 3. shuffle invariance: reorder questions and each question's options
    rng = random.Random(0)
    perm = list(range(len(qs)))
    rng.shuffle(perm)
    shuffled = []
    for j in perm:
        q = dict(qs[j])
        if q["type"] == "choice":
            q["options"] = rng.sample(q["options"], len(q["options"]))
        shuffled.append(q)
    _, res2 = run_packed(shuffled)
    worst = 0.0
    for new_i, old_i in enumerate(perm):
        worst = max(worst, (res2[new_i]["decide"] - res[old_i]["decide"]).abs().max().item())
        old = dict(zip(answers(qs[old_i]), res[old_i]["opts"]))
        for a, ll in zip(answers(shuffled[new_i]), res2[new_i]["opts"]):
            worst = max(worst, abs(old[a] - ll))
    print(f"shuffle invariance: worst |Δ| = {worst:.2e}")
    assert worst < 1e-3, worst
    print("pack.py self-check OK")
