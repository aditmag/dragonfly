<p align="center">
  <img src="assets/dragonfly.png" alt="Dragonfly" width="600">
</p>

<h1 align="center">Query images  in milliseconds</h1>

<h3 align="center">Dragonfly is a LoRA fine-tune of Qwen3-VL with typed output heads that answers many questions about an image in one  pass, as calibrated probabilities.</h3>

<p align="center">
  <a href="https://adit.run/dragonfly"><img src="https://img.shields.io/badge/demo-adit.run%2Fdragonfly-orange" alt="demo: adit.run/dragonfly"></a>
</p>

### How it works

Dragonfly reads the image once and scores every question in the same pass with no text generation, which is why it's fast. Thanks to an attention mask, the answer only depends on the image and the specific question independently, irrespective of how many questions are asked.

<div align="center">

| Questions | Time |
|---:|---:|
| 100 | 0.22 s |
| 250 | 0.82 s |
| 500 | 2.30 s |
| 1,000 | 8.20 s |



NVIDIA GH200, bf16, median of 5 runs.

</div>

### Typed answers

A VLM can reply with anything. Dragonfly can only answer in the form you asked for, with a probability for each
possible answer.

<div align="center">

| Type | Question | Qwen3-VL-4B | Dragonfly |
|---|---|---|---|
| Yes / no | "Is the plane off the ground?" | "Yes, the plane is off the ground. In this historical photograph, the Wright Flyer…" | yes 80%<br>no 20%<br>*can't answer 0%* |
| Choice | "What time of day is it: morning, noon, or night?" | "Based on the visual evidence in the image, it is most likely **morning**. Here's the reasoning: …" | morning 58%<br>noon 30%<br>night 12%<br>*can't answer 35%* |
| Scale | "How cluttered is the scene, from 1 (tidy) to 3 (cluttered)?" | "The scene is **1 (tidy)**. While there are a few scattered objects — like a small crate or box in the foreground…" | 1: 74%<br>2: 19%<br>3: 6%<br>*can't answer 5%* |

</div>

"Can't answer" is the probability that the question doesn't fit the image. When told to pick a colour for the dog (brown, black, or white) in a photo with no dog, Qwen3-VL-4B picked "black"; Dragonfly returns can't answer 98%.

### Data

No dataset asks typed questions with probability answers, so we built one: human-labelled questions from public
VQA datasets, plus questions written and labelled by larger Qwen VLMs, calibrated against the human labels.

<div align="center">

| Source | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Human vote spreads (VQAv2, KonIQ-10k) | 79,105 | 16,005 | 18,031 | 113,141 |
| Exact labels (GQA, A-OKVQA, TallyQA, VizWiz, COCO / Visual Genome) | 174,093 | 30,518 | 43,855 | 248,466 |
| Written and labelled by Qwen3-VL-32B and Qwen2.5-VL-32B | 325,757 | 8,176 | 13,821 | 347,754 |
| Option-count variants | 30,275 | – | – | 30,275 |
| **Questions** | **609,230** | **54,699** | **75,707** | **739,636** |
| **Images** | **123,048** | **9,082** | **9,706** | **141,836** |

</div>

### Run it

```
uv sync
uv run python model/serve.py --run runs/full4b      # then open http://127.0.0.1:8800
```

`runs/full4b` is the trained adapter (`lora/`, `heads.pt`, `log.json`); the 4B and 8B adapters will be published
on Hugging Face. The 4B needs one GPU with ~12 GB (add `--dtype float16` on GPUs without bf16).

### License

Code: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). The data sources keep their own licenses, and images are
not redistributed ([data/IMAGES.md](data/IMAGES.md)).

### Thanks

We acknowledge CSC – IT Center for Science, Finland, for computational resources. Built on Qwen3-VL and
Qwen2.5-VL (Alibaba Qwen team). Data from COCO, Visual Genome, VQAv2, GQA, VizWiz (CC BY 4.0), A-OKVQA, TallyQA
(Apache-2.0), KonIQ-10k, and POPE (MIT, evaluation only).
