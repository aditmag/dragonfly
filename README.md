<p align="center">
  <img src="assets/dragonfly.png" alt="Dragonfly" width="600">
</p>

<h3 align="center">Dragonfly is a LoRA fine-tune of Qwen3-VL with typed output heads that answers many questions about an image in one pass, with a probability for every answer.</h3>

<p align="center">
  <a href="https://adit.run/dragonfly"><img src="https://img.shields.io/badge/demo-adit.run%2Fdragonfly-orange" alt="demo: adit.run/dragonfly"></a>
</p>

### How it works

Dragonfly reads the image once and answers every question in the same forward pass, without generating text. The
questions sit side by side behind the image in one sequence. An attention mask stops them from seeing each other and
they share position ids, so a packed answer matches asking that question alone (checked to ~1e-4 in float32 on the
2B model), and reordering questions or options doesn't change the answers. The heads start from the base model's own
Yes/No and option log-probabilities, so before training Dragonfly answers exactly like Qwen3-VL read that way. LoRA
(rank 16) and the small heads are trained from there.

### Typed answers

A VLM can reply with anything. Dragonfly can only answer in the form you asked for, with a probability for each
possible answer.

<div align="center">

| Type | Question | Qwen3-VL-4B, chat reply | Dragonfly |
|---|---|---|---|
| Yes / no | "Is the plane off the ground?" | "Yes, the plane is off the ground. In this historical photograph, the Wright Flyer…" | yes 80%<br>no 20%<br>*can't answer 0%* |
| Choice | "What time of day is it: morning, noon, or night?" | "Based on the visual evidence in the image, it is most likely **morning**. Here's the reasoning: …" | morning 58%<br>noon 30%<br>night 12%<br>*can't answer 35%* |
| Scale | "How cluttered is the scene, from 1 (tidy) to 3 (cluttered)?" | "The scene is **1 (tidy)**. While there are a few scattered objects — like a small crate or box in the foreground…" | 1: 74%<br>2: 19%<br>3: 6%<br>*can't answer 5%* |

</div>

Examples picked by us from the demo, not a random sample.

"Can't answer" is a separate output: the probability that the question doesn't fit the image. The other
probabilities assume it does. It was trained mostly on questions about objects that aren't in the photo ("What colour
is the dog?" with no dog) and catches those well (AUROC 0.999 on our test set). It doesn't generalise much beyond
that: on ImajevBench it flagged 1 of 24 questions whose answer is "Unknown".

### Results

<div align="center">

| Test set, accuracy (ECE) | Qwen3-VL-4B | Dragonfly 4B | Qwen3-VL-8B | Dragonfly 8B |
|---|---:|---:|---:|---:|
| Yes / no | 85.8% (0.011) | 88.1% (0.007) | 86.4% (0.020) | 88.4% (0.006) |
| Choice | 82.1% (0.010) | 83.9% (0.003) | 82.3% (0.011) | 84.1% (0.008) |
| Scale | 72.7% (0.023) | 77.8% (0.009) | 73.8% (0.031) | 78.2% (0.012) |
| Held out of training: open / closed | 70.6% | 72.3% | 72.0% | 73.2% |
| Held out of training: material | 87.1% | 88.2% | 88.0% | 88.9% |
| POPE adversarial | 88.0% (0.022) | 89.0% (0.031) | 87.5% (0.019) | 88.3% (0.032) |
| ImajevBench, dev + calibration (254 items) | 61.8% | 64.2% | 69.3% | 70.9% |

</div>

- The Qwen3-VL columns are the base model read through its own Yes/No and option log-probabilities, which is
  Dragonfly before training. Every model gets a per-type temperature fitted on validation images.
- Accuracy is the label's probability on the model's top answer (plain accuracy for exact labels). ECE uses 15 bins.
- 18% of test questions were labelled by the Qwen 32B teachers, so on those, accuracy measures agreement with them.
- On our test set, training adds 0.7–5.1 points. Temperature scaling alone already calibrates the base model on yes/no and choice
  (raw ECE 0.11–0.14, 0.010–0.011 after scaling); training helps calibration mainly on scales. On POPE, calibration
  got worse (4B: 0.022 → 0.031).
- ImajevBench (AI-generated images, often with written rules our data never covered) is the only outside benchmark
  we ran. Training adds 2.4 (4B) and 1.6 (8B) points there. imajev-4b reports 83.9% on the benchmark's test
  split, whose labels aren't public, so the two numbers come from different items.
- Raw numbers: [results/](results/).

### Data

No dataset asks typed questions with probability answers, so we built one: human-labelled questions from public
VQA datasets, plus questions written and labelled by larger Qwen VLMs, calibrated against the human labels.

<div align="center">

| Source | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Human vote spreads (VQAv2, KonIQ-10k) | 79,105 | 16,005 | 18,031 | 113,141 |
| Exact labels (GQA, A-OKVQA, TallyQA, VizWiz, COCO / Visual Genome) | 174,093 | 30,518 | 43,855 | 248,466 |
| Written and labelled by Qwen3-VL-32B and Qwen2.5-VL-32B | 325,757 | 8,176 | 13,821 | 347,754 |
| Option-count variants (copies with fewer options) | 30,275 | – | – | 30,275 |
| **Questions** | **609,230** | **54,699** | **75,707** | **739,636** |
| **Images** | **123,048** | **9,082** | **9,706** | **141,836** |

</div>

### Limits

- **Not faster than a well-served VLM yet.** On an RTX 5070 (one image, median of 5), 10 questions take 0.15 s for
  both Dragonfly and batched vLLM with prefix caching. At 100 questions Dragonfly takes 1.22 s and vLLM 0.58 s. The
  dense attention mask grows with the square of the sequence length; about 250 questions fit in one pass on 12 GB.
- **Photos only.** Training images are COCO, Visual Genome, VizWiz and KonIQ photos. Images are downscaled to at
  most 448×448 pixels, so small text and fine detail are lost.
- **No reasoning step.** Nothing is generated, so there is no room to work through a question step by step.

### Run it

```
uv sync
uv run python model/serve.py --run runs/full4b      # then open http://127.0.0.1:8800
```

`--run` takes a training output folder (`lora/`, `heads.pt`, `log.json`). The trained 4B and 8B adapters are not
published yet; they will go on Hugging Face. The 4B needs one GPU with ~12 GB (add `--dtype float16` on GPUs
without bf16).

### License

Code: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). The data sources keep their own licenses, and images are
not redistributed ([data/IMAGES.md](data/IMAGES.md)).

### Thanks

We acknowledge CSC – IT Center for Science, Finland, for computational resources. Built on Qwen3-VL and
Qwen2.5-VL (Alibaba Qwen team). Data from COCO, Visual Genome, VQAv2, GQA, VizWiz (CC BY 4.0), A-OKVQA, TallyQA
(Apache-2.0), KonIQ-10k, and POPE (MIT, evaluation only).
