"""Typed heads on a packed sequence, warm-started so step 0 == the baseline.

Baseline = the frozen LM head's own readout:
  bool    logit = logp(Yes) - logp(No) at DECIDE            -> P(yes) = sigmoid(logit)
  choice  score_k = log P(option_k + <|im_end|> | question)  -> softmax over the options
  score   same as choice, options = the scale values         -> distribution over the scale
Learned terms are added on top and start at exactly 0:
  bool    + w . h_DECIDE                       (w = 0)
  option  + (Q h_DECIDE) . (O h_OPT_k) / sqrt(r)  (O = 0; h_OPT = hidden at the option's <|im_end|>)
  N/A     sigmoid(u . h_DECIDE + b)            (u = 0, b = -4: P(N/A) ~ 0.018, the baseline has no N/A)
So at step 0 the answer distributions equal the baseline exactly; LoRA and these terms move away
from it only where that lowers the loss.
"""
import math

import torch
from torch import nn

NA_BIAS = -4.0


class Heads(nn.Module):
    def __init__(self, hidden, yes_id, no_id, rank=64):
        super().__init__()
        self.yes_id, self.no_id = yes_id, no_id
        self.bool_w = nn.Linear(hidden, 1)
        self.na = nn.Linear(hidden, 1)
        self.q_proj = nn.Linear(hidden, rank, bias=False)
        self.o_proj = nn.Linear(hidden, rank, bias=False)
        for m in (self.bool_w, self.na, self.o_proj):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.bool_w.bias)
        nn.init.constant_(self.na.bias, NA_BIAS)
        self.scale = 1 / math.sqrt(rank)

    def forward(self, h, packed, lm_head):
        """h: (L, H) final-norm hidden states of one packed sequence. lm_head: the model's frozen
        output layer. Returns per question {"na": logit, "bool": logit} or {"na": logit, "opt": (K,)}."""
        # every row whose next-token log-probs we need: DECIDE rows, and inside option blocks
        rows, targets, owner = [], [], []  # owner: (question, option) the log-prob belongs to
        for i, (d, opts) in enumerate(zip(packed.decide, packed.options)):
            if not opts:
                rows += [d, d]; targets += [self.yes_id, self.no_id]; owner += [(i, "yes"), (i, "no")]
            for k, (s, e, toks) in enumerate(opts):
                rows += [d] + list(range(s, e - 1)); targets += toks; owner += [(i, k)] * len(toks)
        rows_t = torch.tensor(rows, device=h.device)
        logp = lm_head(h[rows_t]).float().log_softmax(-1)
        picked = logp.gather(1, torch.tensor(targets, device=h.device)[:, None])[:, 0]

        hd = h[torch.tensor(packed.decide, device=h.device)].float()  # heads stay fp32 under a bf16 model
        na = self.na(hd)[:, 0]
        bool_extra = self.bool_w(hd)[:, 0]
        qv = self.q_proj(hd)

        out = [{"na": na[i]} for i in range(len(packed.decide))]
        sums = {}
        for (i, k), lp in zip(owner, picked):
            sums[(i, k)] = sums.get((i, k), 0) + lp
        for i, opts in enumerate(packed.options):
            if not opts:
                out[i]["bool"] = sums[(i, "yes")] - sums[(i, "no")] + bool_extra[i]
                continue
            ll = torch.stack([sums[(i, k)] for k in range(len(opts))])
            ho = h[torch.tensor([e - 1 for _, e, _ in opts], device=h.device)].float()
            out[i]["opt"] = ll + (self.o_proj(ho) @ qv[i]) * self.scale
        return out


def hidden_states_batch(model, packs, entries, dtype, pad_id):
    """Final-norm hidden states for several packed sequences in one forward: right-padded to the longest,
    each keeping its own isolation mask and positions; padding rows attend only to themselves (no NaN)
    and nothing attends to them. Vision from cache-style entries (cache.encode_pixels): Qwen3-VL
    concatenates per-image features in batch order and scatters them into the image tokens row-major,
    i.e. image 0's tokens first -- the same order. Returns [(L_b, H)] trimmed to each sequence.
    One image per forward used the GH200 at ~20% (exit run: 0.39 s/image vs ~0.1 s of arithmetic)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures
    dev = model.device
    L = max(len(p.input_ids) for p in packs)
    B = len(packs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    pos = torch.zeros((3, B, L), dtype=torch.long)
    allowed = torch.zeros((B, L, L), dtype=torch.bool)
    for b, p in enumerate(packs):
        n = len(p.input_ids)
        ids[b, :n], pos[:, b, :n], allowed[b, :n, :n] = p.input_ids, p.position_ids, p.allowed
        allowed[b, torch.arange(n, L), torch.arange(n, L)] = True
    mask = torch.zeros((B, 1, L, L), dtype=dtype).masked_fill(~allowed[:, None], torch.finfo(dtype).min)
    mm = BaseModelOutputWithDeepstackFeatures(
        pooler_output=tuple(e["embeds"].to(dev, dtype) for e in entries),
        deepstack_features=[tuple(e["deepstack"][k].to(dev, dtype) for e in entries) for k in range(len(entries[0]["deepstack"]))],
    )
    h = model.model(input_ids=ids.to(dev), position_ids=pos.to(dev), attention_mask=mask.to(dev),
                    mm_encoder_outputs={"image": mm},
                    image_grid_thw=torch.stack([e["grid"] for e in entries]).to(dev)).last_hidden_state
    return [h[b, :len(p.input_ids)] for b, p in enumerate(packs)]


def hidden_states(model, packed, dtype, **vision):
    """Final-norm hidden states (L, H) of one packed sequence. vision: pixel_values + image_grid_thw,
    or mm_encoder_outputs (cached)."""
    dev = model.device
    return model.model(
        input_ids=packed.input_ids[None].to(dev),
        position_ids=packed.position_ids[:, None].to(dev),
        attention_mask=packed.mask(dtype).to(dev),
        **vision,
    ).last_hidden_state[0]


if __name__ == "__main__":
    # Step-0 check: freshly initialised heads on the packed sequence ==
    # the baseline computed from separate stock-HF runs of each question.
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from pack import END, answers, check_record, check_setup, pack, prompt_for

    MODEL, root, img_dir, dev = check_setup()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    tok = proc.tokenizer
    SIZE = {"longest_edge": 448 * 448, "shortest_edge": 64 * 64}
    heads = Heads(model.config.text_config.hidden_size, tok.convert_tokens_to_ids("Yes"),
                  tok.convert_tokens_to_ids("No")).to(dev)

    image_id, qs = check_record(root, img_dir, skip=1)  # a different record from pack.py's check
    image = Image.open(img_dir / f"{image_id}.jpg").convert("RGB")
    vis = proc.image_processor(images=[image], size=SIZE, return_tensors="pt").to(dev)
    p = pack(qs, vis["image_grid_thw"][0], tok, model.config.image_token_id)
    with torch.no_grad():
        out = heads(hidden_states(model, p, torch.float32, **vis), p, model.lm_head)

    worst = 0.0
    for i, q in enumerate(qs):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_for(q)}]}]
        chat = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inp = proc(text=[chat], images=[image], images_kwargs={"size": SIZE}, return_tensors="pt").to(dev)
        with torch.no_grad():
            lp = model(**inp).logits[0, -1].float().log_softmax(-1).cpu()
        if q["type"] == "bool":
            base = torch.sigmoid(lp[heads.yes_id] - lp[heads.no_id])
            ours = torch.sigmoid(out[i]["bool"]).cpu()
            d = (base - ours).abs().item()
            desc = f"P(yes) baseline {base:.4f} ours {ours:.4f}"
        else:
            n, lls = inp["input_ids"].shape[1], []
            for a in answers(q):
                inp_o = proc(text=[chat + a + END], images=[image], images_kwargs={"size": SIZE},
                             return_tensors="pt").to(dev)
                with torch.no_grad():
                    lpo = model(**inp_o).logits[0].float().log_softmax(-1).cpu()
                lls.append(sum(lpo[n - 1 + k, t] for k, t in enumerate(inp_o["input_ids"][0, n:].tolist())))
            base = torch.stack(lls).softmax(0)
            ours = out[i]["opt"].softmax(0).cpu()
            d = (base - ours).abs().max().item()
            desc = f"argmax baseline {answers(q)[base.argmax()]!r} ours {answers(q)[ours.argmax()]!r}"
        worst = max(worst, d)
        print(f"  q{i} {q['type']:6s} max|Δp|={d:.1e}  P(N/A)={torch.sigmoid(out[i]['na']).item():.3f}  {desc}")
    print(f"step-0 vs baseline: worst |Δ probability| = {worst:.2e}")
    assert worst < 1e-3, worst
    print("heads.py step-0 check OK")

    # batched forward (training) == one image at a time: 3 records of different lengths, padded together
    from cache import encode, mm_inputs
    recs = [check_record(root, img_dir, skip=s) for s in (1, 2, 3)]
    imgs = [Image.open(img_dir / f"{i}.jpg").convert("RGB") for i, _ in recs]
    entries = encode(model, proc.image_processor, imgs)
    packs = [pack(q, e["grid"], tok, model.config.image_token_id) for (_, q), e in zip(recs, entries)]
    with torch.no_grad():
        hb = hidden_states_batch(model, packs, entries, torch.float32, tok.pad_token_id)
        worst = 0.0
        for p, e, h_b in zip(packs, entries, hb):
            h_1 = hidden_states(model, p, torch.float32, **mm_inputs(e, dev, torch.float32))
            for o1, o2 in zip(heads(h_1, p, model.lm_head), heads(h_b, p, model.lm_head)):
                worst = max(worst, max((o1[k] - o2[k]).abs().max().item() for k in o1))
    print(f"batched vs single ({[len(p.input_ids) for p in packs]} tokens): worst |Δ logit| = {worst:.2e}")
    assert worst < 1e-3, worst
    print("heads.py batch check OK")
