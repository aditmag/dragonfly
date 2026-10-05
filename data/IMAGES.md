# Images

Building the dataset needs no images. They're only needed for the teacher, training, evaluation and the
demo, and they are **not redistributed** here (COCO/Visual Genome photos keep their Flickr licences).
`fetch_images.py` downloads exactly the images a manifest lists, from the original hosts.

## Where they go

`$DF_DATA/images/<image_id>.jpg` (default `data/images`): the original JPEG bytes, unmodified. EXIF rotation
and RGB conversion happen at load time, so re-fetching never applies them twice. The ~142k selected images
take ~21 GB.

## How to run it

```
uv run python data/fetch_images.py --manifest data/out/images_selected.jsonl [--limit 50] [--workers 24]
```

- `--manifest`: `images.jsonl` (full raw pool) or `images_selected.jsonl` (the training set from
  `selection.py`). Use `images_selected.jsonl` normally.
- `--limit N`: stop after N images (smoke test). Omit for a full run.
- Re-running skips images already on disk (checked by file name).
- Dead links go to `data/out/missing.jsonl` (`image_id`, `reason`) instead of stopping the run.

## Per-source rules

- **COCO, Visual Genome** (`url` field): one GET per image, direct. Most images are these two sources.
- **VizWiz, KonIQ** (`archive` + `member` fields): these only ship as zips. VizWiz needs a few hundred
  members of an 11 GB archive, so they're read with HTTP range requests through `zipfile` (stdlib only).
  KonIQ needs nearly all of its 767 MB archive (`koniq10k_512x384.zip`), so it's downloaded once and read
  locally; 512×384 fits the 448×448 pixel budget without resizing.
- Every request has a timeout and retries with backoff; one hung request can't stall the run.
