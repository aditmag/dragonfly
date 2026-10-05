# Dragonfly (research preview)

Ask an image many typed questions and get a calibrated probability distribution for each one, from **one
forward pass**, with no text generation.

**Demo:** https://adit.run/dragonfly · **Write-up:** [docs/REPORT.md](docs/REPORT.md)

| Type | Example | Output |
|---|---|---|
| yes / no | "Is the plane off the ground?" | P(yes), P(can't answer) |
| choice | "How long did this flight last? [about 12 seconds, about 12 minutes, …]" | a distribution over your options, P(can't answer) |
| scale | "How cluttered is the scene? 1 = very tidy … 5 = very cluttered" | a distribution over the scale, P(can't answer) |

"Can't answer" is its own probability: the question doesn't fit the image (e.g. it asks about something that
isn't there).

The idea comes from TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a
"System One" model for text. Jev's internals are unpublished; this is an independent design for images.

## How it works

- **Base:** Qwen3-VL-4B-Instruct (also trained: 8B). The vision encoder stays frozen; the LLM gets LoRA (r = 16).
- **One packed sequence:** the image is encoded once, then every question (and every choice option) is appended
  behind it. An isolation mask and per-block positions mean questions can't see each other, so their order,
  and the order of options, changes the outputs by exactly 0 (checked to ~1e-4). Each extra question costs
  ~10 tokens, not another image.
- **Typed heads:** read the hidden states at positions the sequence already has. They're warm-started so that,
  before training, the model *is* the base model's own answer readout; training only moves away from it where
  that lowers the loss.
- **Training:** proper scoring rules (BCE, cross-entropy, ranked probability score for scales), then one
  temperature per answer type.
- **Data:** 739,636 questions on 141,836 images: human-labelled VQA sets plus 348k questions written and labelled
  by larger Qwen VLMs, calibrated against human labels ([DATA.md](DATA.md)).

## Results

9,706 held-out test images. Accuracy / calibration error (ECE, lower is better):

| | 4B base | **4B trained** | 8B base | **8B trained** |
|---|---|---|---|---|
| yes / no (41.8k) | .858 / .011 | .881 / .007 | .864 / .020 | **.884 / .006** |
| choice (15.2k) | .821 / .010 | .839 / **.003** | .823 / .011 | **.841** / .008 |
| scale (16.5k) | .727 / .023 | .778 / **.009** | .738 / .031 | **.782** / .012 |
| held-out families: material / open-closed | .871 / .706 | .882 / .723 | .880 / .720 | **.889 / .732** |
| POPE adversarial (yes-bias) | .880 | **.890** | .875 | .883 |
| "can't answer" AUROC | 0.50 | 0.999 | 0.50 | 0.999 |
| ECE with no temperature | .11–.18 | ≤ .012 | .10–.14 | ≤ .011 |

- Training beats the base model on every question type, including two question families never seen in training.
- Training matters more than size: the trained 4B beats the untrained 8B everywhere.
- Speed: 4 questions in one pass take ~60–90 ms of model time on a single GPU (RTX 5070 / GH200).
- Limits: no reasoning step, so counting past ~5, small text and arithmetic are weak. On
  [ImajevBench](https://huggingface.co/datasets/mohit67890/imajev-bench) (synthetic menus, receipts and signs, often
  with written rules; its labelled dev + calibration splits) it scores 64% (4B) / 71% (8B), well behind models
  trained for that task. Details and caveats in [docs/REPORT.md](docs/REPORT.md).

Raw numbers: [results/](results/).

## Run it

```
uv sync
uv run python model/serve.py --run runs/full4b      # then open http://127.0.0.1:8800
```

`runs/full4b` holds the trained adapter (`lora/`, `heads.pt`, `log.json`); the trained 4B and 8B adapters will
be published on Hugging Face. One GPU with ~12 GB is enough for the 4B in bf16 (`--dtype float16` on GPUs
without bf16).

## Reproduce

Everything is deterministic given the stored teacher outputs; see [docs/REPORT.md](docs/REPORT.md) for the
reasoning behind each step.

1. **Human-labelled base** (CPU, standard library). Download the annotations listed in [DATA.md](DATA.md) into
   `data/raw/<source>/`, then:
   ```
   for s in coco vg vqav2 vqav2_choice gqa aokvqa tallyqa koniq vizwiz templates pope; do uv run python data/$s.py; done
   uv run python data/merge.py && uv run python data/selection.py
   mkdir -p data/out/v2 && cp data/out/dataset_selected.jsonl data/out/v2/base_selected.jsonl
   uv run python data/fetch_images.py --manifest data/out/images_selected.jsonl     # see data/IMAGES.md
   ```
2. **Teacher-written questions** (GPU, vLLM 0.29.0, two 32B VLMs; ~20 GPU-hours on GH200s for all images):
   ```
   uv run python data/topics.py                       # question specs per image -> data/out/v2/specs.jsonl
   uv run python data/teacher.py calibset             # human-labelled calibration questions
   uv run python data/teacher.py calibrate --model Qwen/Qwen3-VL-32B-Instruct   --out data/out/v2/cal_q3.json
   uv run python data/teacher.py calibrate --model Qwen/Qwen2.5-VL-32B-Instruct --out data/out/v2/cal_q25.json
   uv run python data/teacher.py write    --out data/out/v2/written.jsonl          # add --shard i/n to split
   uv run python data/teacher.py classify --inputs data/out/v2/written.jsonl --out data/out/v2/verdicts.jsonl
   uv run python data/teacher.py label --types bool --inputs data/out/v2/written.jsonl \
       --verdicts data/out/v2/verdicts.jsonl --calibration data/out/v2/cal_q3.json --out data/out/v2/lab_q3.jsonl
   uv run python data/teacher.py label --model Qwen/Qwen2.5-VL-32B-Instruct --types choice score \
       --inputs data/out/v2/written.jsonl --verdicts data/out/v2/verdicts.jsonl \
       --calibration data/out/v2/cal_q25.json --out data/out/v2/lab_q25.jsonl
   uv run python data/teacher.py recalibrate --calibration data/out/v2/cal_q25.json \
       --rots data/out/v2/cal_q25.premise.rots.json --labelled data/out/v2/lab_q25.jsonl --out data/out/v2/cal3_q25.json
   uv run python data/teacher.py finalize --stability 0.03 \
       --inputs data/out/v2/lab_q3.jsonl:data/out/v2/cal_q3.json:bool data/out/v2/lab_q25.jsonl:data/out/v2/cal3_q25.json:choice,score
   uv run python data/merge.py && uv run python data/selection.py && uv run python data/qa.py data/out/dataset_selected.jsonl
   ```
3. **Train and evaluate** (GPU; ~2 h for the 4B on one GH200):
   ```
   uv run python model/train.py --model Qwen/Qwen3-VL-4B-Instruct --train-ids data/out/ids_train.txt \
       --val-ids data/out/ids_val.txt --images data/images --out runs/full4b --epochs 1 --accum 16 --batch 8 \
       --warmup 200 --final-val-limit 2000 --workers 16
   uv run python model/evaluate.py --ids data/out/ids_test.txt --val-ids data/out/ids_val.txt --val-limit 1000 \
       --images data/images --adapter runs/full4b --out runs/full4b/eval_test.json
   ```
   Self-checks against the real model (packed == separate, order invariance, step 0 == base):
   `python model/pack.py`, `python model/heads.py`, `python model/cache.py`.

## Licence

Code: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Data sources keep their own licences; images are not
redistributed ([DATA.md](DATA.md)).

## Acknowledgements

We acknowledge CSC – IT Center for Science, Finland, for computational resources. Built on Qwen3-VL and
Qwen2.5-VL (Alibaba Qwen team) and on the datasets listed in [DATA.md](DATA.md).
