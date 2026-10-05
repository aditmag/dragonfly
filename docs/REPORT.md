# Dragonfly: typed questions about an image, answered in one forward pass

*Research preview, October 2026.*

Dragonfly takes one image and any number of typed questions (yes/no, multiple choice, or a numeric scale) and returns
a calibrated probability distribution for every question from a single forward pass of a vision-language model. It
never generates text. Every answer also gets a separate probability that the question can't be answered from the
image.

The goal was to test whether the "System One" interface of TypeSafe's text-only Jev (typed questions in, calibrated
probabilities out, all questions evaluated in parallel and in isolation) can be built for images on top of an open
VLM, and whether it can be made both faster and better calibrated than the VLM it starts from.

## 1. Interface

| Type | Input | Output |
|---|---|---|
| yes / no | a question | P(yes), P(can't answer) |
| choice | a question and 2–8 options (any order) | a distribution over the options, P(can't answer) |
| scale | a question, a range (e.g. 1–5) and what the ends mean | a distribution over the range, P(can't answer) |

"Can't answer" is for questions that don't fit the image, typically because they presuppose something that isn't
there ("What colour is the dog?" with no dog). "No" stays "no": "Is there a dog?" with no dog is answerable.

## 2. Model

**Base.** Qwen3-VL-4B-Instruct (and 8B for a size comparison). The vision encoder and its merger stay frozen; the
language model gets LoRA (rank 16) on every attention and MLP projection.

**Packing.** The image is encoded once (~190 tokens at a 448×448 pixel budget). Every question, and for choice and
scale questions every option, becomes a short block appended after the image in one sequence:

- an **isolation mask** lets each block see the image and itself, nothing else (an option block also sees its own
  question);
- **positions restart** at the same value for every question block (per-block M-RoPE positions), so no question is
  "later" than another.

The outputs therefore can't depend on which other questions were asked or in what order, and option order can't
matter either. Each extra question costs its own ~10–30 tokens, not another copy of the image.

**Heads.** They read the final hidden state at positions the sequence already has (the end of a question, the end of
an option); there is no new vocabulary.

- yes / no: log p("Yes") − log p("No") from the frozen LM head, plus a learned linear term;
- choice and scale: each option's log-likelihood under the LM, plus a learned rank-64 bilinear term between the
  question and option states;
- can't answer: a sigmoid on the question state, initialised to ~2%.

The learned terms start at zero, so **before training the model is exactly the base model's own answer readout**.
Training moves away from it only where that lowers the loss.

**Correctness checks** (`model/pack.py`, `heads.py`, `cache.py`, run against the real models):

| | 4B | 8B |
|---|---|---|
| packed vs each question run alone (max abs. diff. in log-probability) | 1.2e-4 | 8.0e-5 |
| shuffling questions and options (max abs. diff.) | 8.8e-5 | 6.5e-5 |
| untrained heads vs the base model (max abs. diff. in probability) | 2.9e-9 | 6.4e-6 |
| batched vs one image at a time (max abs. diff. in logit) | 7.3e-5 | 7.0e-5 |

## 3. Data

No dataset asks typed questions with calibrated answer distributions, so it was built (739,636 questions on 141,836
images; sources and licences in [DATA.md](../DATA.md)).

**Human-labelled base.** VQAv2 yes/no questions, and short-answer VQAv2 questions turned into multiple choice (with
the spread of the 10 human answers as the target), GQA verify questions, A-OKVQA multiple choice, TallyQA counts as a 0–9 scale, KonIQ-10k quality (human
vote histograms), VizWiz unanswerable questions, and COCO / Visual Genome templates (presence with hard negatives,
relations, colours, and "can't answer" questions about COCO objects confirmed absent). Train is capped per source so
no high-volume source dominates.

**Why a teacher.** The base averages ~2 questions per training image, nearly all choice questions have exactly 4
options, there are only two kinds of scale, and topics like weather, condition, emotion or lighting barely appear. A
first generated version (v1) made this clear in the other direction: a dedup bug starved a third of the images, the
generated yes/no labels said "yes" 73% of the time, a third of them were either/or questions, choice options
included "cannot determine", and the answer leaned to the first option. Version 2 was rebuilt from scratch with an
automated acceptance check for each of these problems.

**Writer.** Every image gets deterministic *specs* (seeded by its id) from a 34-topic taxonomy: question type, topic,
exact option count (2–8), scale and anchor style (5 scales, 20% each), an intended yes/no answer (50/50, so "no"
questions are plausible-but-false rather than rare), and, for 22%, an absent object for a "can't answer" question.
Qwen3-VL-32B-Instruct sees the image, its existing questions and the spec, and writes the question under a JSON schema
built from the spec. The schema fixes the option count and forces the first word (choice questions can't be yes/no
questions in disguise). Rule-based checks reject answer leaks, "unknown"-style options, either/or yes/no questions and
near-duplicates.

**Labelling.** Each question is asked with lettered options in several rotated orders, plus an explicitly worded
"cannot be answered from this image (what it asks about is not shown, or its premise is false)" option that rotates
with the others. The answer is read from the model's probabilities over the offered letters (the first token,
restricted to those letters): no generated text. Two teachers were compared against 8,820 held-out human labels;
the best per type was kept (*hybrid*):

| teacher (after temperature + N/A bias) | yes/no ECE / acc. | choice ECE / acc. | scale ECE / acc. |
|---|---|---|---|
| Qwen3-VL-32B-Instruct | **0.033 / 0.889** | 0.030 / 0.877 | 0.068 / 0.612 |
| Qwen2.5-VL-32B-Instruct | 0.043 / 0.850 | **0.020 / 0.881** | **0.028 / 0.593** |

So Qwen3-VL-32B labels yes/no and Qwen2.5-VL-32B labels choice and scale. This also means the writer doesn't grade
most of its own questions; on yes/no, where it does, its agreement with its own intended answers was no higher than the
other model's (no sign of self-preference). "Can't answer" detection: AUROC 0.975–0.982.

**Bias correction.** After the full run, the most likely answer still landed on some option positions more than
chance (5.5 percentage points off uniform; with 7 options the last won 22% instead of 14%). Two causes: the teacher
prefers some letters, and the option next to the rotating "can't answer" option is favoured. Following PriDe (Zheng
et al., ICLR 2024), a per-letter bias is estimated from all stored readouts and removed, and a per-position prior is
divided out (separately for answerable and "can't answer" questions). Residual: within 1.0 point of uniform overall.
On human labels the correction is neutral.

**Stability filter.** Questions whose rotated readouts disagree (Jensen–Shannon divergence > 0.03 across orders)
depend on option order rather than the image and are dropped (in the final pilot: ~1% of yes/no, ~13% of choice
and ~4% of scale questions; calibration showed accuracy around 0.6 beyond that threshold).

**Determinism.** All randomness is a hash of the image id. With vLLM's batch-invariant mode, every step after the
written questions reproduces bit for bit; the sampled writer differs on ~1.8% of specs across separate runs, so its
output is the stored artifact. Rebuilding the dataset from the stored outputs is byte-identical.

**Result.** 347,754 teacher questions (yes/no 147k, choice 142k, scale 59k), 113,141 with human vote spreads and
248,466 with exact labels, plus 30,275 option-count variants of human choice questions: 4.71 distinct questions per
training image. All 27 acceptance checks pass (`data/qa.py`): teacher yes-rate 49%, no either/or yes/no questions, no
"unknown" options, option counts and topics as specified, no 4-word opening above 1%, held-out test families
≥ 2,500 questions each.

## 4. Training

One epoch over 123,048 training images (7,691 steps of 16 images; all questions of an image in one packed sequence),
bf16, LoRA learning rate 1e-4, head learning rate 1e-3, 200 warmup steps then cosine to 10%. Losses are proper
scoring rules against the (soft) targets: BCE for yes/no and "can't answer", cross-entropy for choice,
cross-entropy plus a ranked probability score for scales (so near misses cost less). Answer losses are weighted by
the probability that the question is answerable. Afterwards one temperature per answer type is fitted on validation
images.

| | 4B | 8B |
|---|---|---|
| trainable parameters | 33.0M LoRA + 0.33M heads | 43.6M LoRA + 0.53M heads |
| time on one GH200 | ~2.2 h | ~3.0 h |
| validation loss, before → after | 1.234 → **0.412** | 1.027 → **0.405** |
| fitted temperatures (yes/no, choice, scale) | 1.11, 1.00, 1.05 | 1.05, 1.00, 1.05 |

The temperatures are close to 1: training with proper scoring rules already produced calibrated outputs.

## 5. Results

9,706 held-out test images. Accuracy is the target's probability mass on the predicted answer; ECE uses 15 bins;
each model gets its own temperatures fitted on 1,000 validation images. Cells are accuracy / ECE / NLL.

| | n | 4B base | 4B trained | 8B base | 8B trained |
|---|---|---|---|---|---|
| test loss | | 1.208 | 0.397 | 0.949 | 0.391 |
| yes / no | 41,832 | .858 / .011 / .335 | .881 / .007 / .280 | .864 / .020 / .335 | **.884 / .006 / .276** |
| choice | 15,155 | .821 / .010 / .484 | .839 / **.003** / .406 | .823 / .011 / .481 | **.841** / .008 / **.400** |
| scale | 16,458 | .727 / .023 / .752 | .778 / **.009** / .564 | .738 / .031 / .722 | **.782** / .012 / **.557** |
| held-out: material | 5,085 | .871 | .882 | .880 | **.889** |
| held-out: open / closed | 2,655 | .706 | .723 | .720 | **.732** |
| POPE adversarial / popular / random | 3,000 each | .880 / .891 / .906 | **.890 / .902 / .915** | .875 / .889 / .910 | .883 / .901 / .915 |
| KonIQ quality (human histograms) | 484 | .449 | .576 | .464 | .577 |
| "can't answer" AUROC | | 0.50 | 0.999 | 0.50 | 0.999 |
| ECE without temperature (yes/no, choice, scale) | | .109 / .144 / .181 | .012 / .003 / .006 | .099 / .133 / .140 | .011 / .005 / .008 |

- **Training helps everywhere**, in accuracy, calibration and NLL, including on the two question families held out of
  training entirely: the model generalises rather than memorising question forms.
- **Training beats size.** Training adds 2–5 points; going from 4B to 8B adds 0.2–1.4. The trained 4B beats the
  untrained 8B on every row; the trained 8B leads the trained 4B by under a point.
- **Calibrated out of the box.** Calibration error without any temperature drops from 0.10–0.18 to about 0.01.
- **fp16 = bf16.** On 2,000 test images the trained 4B in fp16 matches bf16 (loss 0.4089 vs 0.4090, accuracies within
  0.1 points), so GPUs without bf16 can serve it.
- **Latency.** With the image already encoded, 4 questions take 61 ms of model time on an RTX 5070 and 76–89 ms on a
  GH200; encoding a new image takes ~50–65 ms. A generating VLM answering the same questions one by one needs a full
  decode per answer.

### ImajevBench

[ImajevBench v2.0-lite](https://huggingface.co/datasets/mohit67890/imajev-bench) asks typed questions about
AI-generated images of menus, receipts, signs and shelves, often together with a written rule ("approve the order if
its total is at most 7.10"); "unknown" means the evidence doesn't decide. Its test labels are withheld, so these are
its labelled dev + calibration splits (254 items), scored with the benchmark's own scorer
([results/imajevbench.json](../results/imajevbench.json)):

| | accuracy | text only | image | image + rule | unknown found |
|---|---|---|---|---|---|
| Qwen3-VL-4B base | 61.8% | 22/38 | 78/103 | 57/113 | 0/24 |
| Dragonfly 4B | 64.2% | 22/38 | 81/103 | 60/113 | 1/24 |
| Qwen3-VL-8B base | 69.3% | 19/38 | 90/103 | 67/113 | 0/24 |
| Dragonfly 8B | 70.9% | 23/38 | 88/103 | 69/113 | 1/24 |

This is far from the models built for that benchmark (its own imajev-4b reports 83.9% on the test split), and the
reasons are clear: our "can't answer" means "the image doesn't fit the question", not "the written evidence is
insufficient"; the training data has no written rules or state; text-only items have no image; and all our images
are photographs. Training still helps a little, and calibration improves a lot (ECE 0.35 → 0.15 on dev for the 4B).

## 6. Limitations

- **No reasoning step.** Counting beyond ~5, reading small text, arithmetic and multi-step spatial reasoning are weak,
  by design.
- **Photographs only.** Documents, screenshots, charts and diagrams are outside the training data.
- **Partly circular evaluation.** About 14% of the test labels are teacher-made; scoring against them measures
  agreement with the teacher. The human and exact labels (86%) are the independent part; a small human-labelled gold
  set for the generated question forms is still to do.
- **"Can't answer" AUROC is optimistic.** Most "can't answer" test questions are templated (a COCO object confirmed
  absent), which is easier than the general case.
- **POPE calibration** gets slightly worse after training (ECE ~0.03 vs ~0.02) even though accuracy rises.
- **The base model knows famous images.** On a well-known photo it can answer knowledge questions (who, where,
  how long) that it couldn't answer from pixels alone.

## 7. Compute

All training and labelling ran on single NVIDIA GH200 GPUs: ~20 GPU-hours for the teacher pipeline, ~2.2 h for the 4B
and ~3 h for the 8B.

## References

- TypeSafe, *Introducing System One models and Jev* (2026). https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Qwen team, Qwen3-VL and Qwen2.5-VL technical reports.
- Zheng et al., *Large Language Models Are Not Robust Multiple Choice Selectors* (PriDe), ICLR 2024.
- Li et al., *Evaluating Object Hallucination in Large Vision-Language Models* (POPE), EMNLP 2023.
- Goyal et al. (VQAv2), Krishna et al. (Visual Genome), Hudson & Manning (GQA), Schwenk et al. (A-OKVQA),
  Acharya et al. (TallyQA), Hosu et al. (KonIQ-10k), Gurari et al. (VizWiz), Lin et al. (COCO).

## Acknowledgements

We acknowledge CSC – IT Center for Science, Finland, for computational resources.
