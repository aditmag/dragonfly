# Data

The training data is built from public annotations plus questions written and labelled by open VLMs.
**No images are redistributed**: records point at image ids, and `data/fetch_images.py` downloads the images from
their original hosts (see [data/IMAGES.md](data/IMAGES.md)).

## Sources

| Source | Used for | Labels | Licence |
|---|---|---|---|
| [COCO](https://cocodataset.org) 2017 annotations | image index, presence / absence and N/A templates | exact | CC BY 4.0 (images: Flickr licences, not redistributed) |
| [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/) 1.2 | relation, colour, material templates | exact | CC BY 4.0 |
| [VQAv2](https://visualqa.org) | yes/no questions, and short-answer questions turned into multiple choice | human vote spread | CC BY 4.0 |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/) balanced | verify (yes/no) questions | exact | CC BY 4.0 |
| [A-OKVQA](https://allenai.org/project/a-okvqa) | 4-way choice | exact | Apache-2.0 |
| [TallyQA](https://github.com/manoja328/TallyQA_dataset) | counts as a 0–9 scale | exact | Apache-2.0 |
| [KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html) | image quality, 1–5 scale | human vote histogram | "freely available to the research community" |
| [VizWiz](https://vizwiz.org) | unanswerable questions (N/A) | exact | CC BY 4.0 |
| [POPE](https://github.com/AoiDragon/POPE) | evaluation only (yes-bias) | exact | MIT |

## Teacher-written questions

348k questions were written by **Qwen3-VL-32B-Instruct** from per-image specs (type, topic, exact option count or
scale, intended answer, or an absent object for "can't answer" questions) and labelled from the models' answer
probabilities over several option orders: yes/no by Qwen3-VL-32B-Instruct, choice and scale by
**Qwen2.5-VL-32B-Instruct**. Both are Apache-2.0. The labels are calibrated against human labels and corrected for
answer-letter and position bias; unstable labels are dropped (`data/teacher.py`).

## The dataset

739,636 questions on 141,836 images (train 123,048 / val 9,082 / test 9,706 images, split by a hash of the image
id; POPE images only in test; two question families, `material` and `open_closed`, only in test). One JSON line per
image:

```json
{"image_id": "coco_000000393221", "split": "train", "questions": [
  {"type": "bool", "q": "Is the hiker walking uphill?", "target": {"yes": 0.31, "no": 0.69}, "na": 0.77,
   "source": "vlm_v2", "label_kind": "teacher_soft", "family": "motion", "kind": "na_bool", "premise": "hiker"}]}
```

`target` is the answer distribution given the question is answerable, `na` the probability it isn't (`target` is
`null` when `na` is 1). `label_kind` is `exact`, `human_soft` or `teacher_soft`; about 14% of test labels are
teacher-made, so report human/exact and teacher-labelled results separately.

The dataset will be published on Hugging Face with the same licence terms as its sources.
