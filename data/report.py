"""Side-by-side v1 vs v2 dataset report -> one self-contained HTML file (dataset v2).

    uv run python data/report.py --v1 out/v1/dataset_selected.jsonl --v2 out/dataset_selected.jsonl \
        --qa1 out/v1/qa_v1.json --qa2 out/v2/qa_v2.json --images images --out out/v2/report.html [--n 300]

A seeded sample of images (stratified by split x image source, only images present in both files),
each with a thumbnail, v1's questions and labels next to v2's, plus per-rotation teacher readouts and
flags; above it the qa.py before/after table. Same inputs -> byte-identical file (no timestamps, sorted
everything, thumbnails re-encoded with fixed settings).
"""
import argparse
import base64
import html
import io
import json
import math
from collections import defaultdict
from pathlib import Path

from qa import EITHER_OR, NA_OPTION, is_aug
from topics import h


def load(path):
    return {r["image_id"]: r for r in map(json.loads, open(path))}


def thumb(path, size=224):
    from PIL import Image
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return ""
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70, optimize=False, progressive=False)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fmt_dist(q):
    t = q.get("target")
    if t is None:
        return "—"
    if isinstance(t, list):
        lo = q["scale"][0]
        pairs = [(str(lo + i), p) for i, p in enumerate(t)]
    else:
        pairs = sorted(t.items(), key=lambda kv: -kv[1])
    return " · ".join(f"{html.escape(k)} {p:.2f}" for k, p in pairs if p >= 0.01) or "—"


def flags(q):
    f = []
    if q["type"] == "bool" and EITHER_OR.search(q["q"]) and not q.get("yesno_ok"):
        f.append("either/or?")
    if q["type"] == "choice" and any(NA_OPTION.search(o) for o in q["options"]):
        f.append("N/A-like option")
    if q["type"] == "choice" and any(o[:1].isupper() for o in q["options"]):
        f.append("Title Case")
    if q.get("stability", 0) > 0.1:
        f.append(f"unstable {q['stability']:.2f}")
    return f


def q_html(q):
    meta = [q["type"], q.get("topic") or q.get("family", ""), q["source"], q["label_kind"].replace("_", " ")]
    if q["type"] == "choice":
        meta.append(f"{len(q['options'])} options")
    if q["type"] == "score":
        meta.append(f"{q['scale'][0]}–{q['scale'][1]} ({html.escape(' / '.join(q['anchors']))})")
    if q.get("premise"):
        meta.append(f"absent: {q['premise']}")
    if q.get("labeller"):
        meta.append(f"labelled by {q['labeller']}")
    rows = [f'<div class="q{" aug" if is_aug(q) else ""}"><div class="qt">{html.escape(q["q"])}</div>',
            f'<div class="meta">{" · ".join(html.escape(str(m)) for m in meta)}{" · copy" if is_aug(q) else ""}</div>']
    if q["type"] == "choice":
        rows.append(f'<div class="opts">{html.escape(" | ".join(q["options"]))}</div>')
    na = q.get("na", 0)
    rows.append(f'<div class="lab">{fmt_dist(q)}<span class="na{" hi" if na >= 0.5 else ""}">N/A {na:.2f}</span></div>')
    for r in q.get("rots", []):
        rows.append('<div class="rot">' + " ".join(f"{html.escape(lab)}:{math.exp(x):.2f}" for lab, x in zip(r["order"], r["lp"])) + "</div>")
    fl = flags(q)
    if fl:
        rows.append('<div class="flags">' + "".join(f"<span>{html.escape(x)}</span>" for x in fl) + "</div>")
    return "".join(rows) + "</div>"


def sample(v1, v2, n):
    both = sorted(set(v1) & set(v2))
    strata = defaultdict(list)
    for i in both:
        strata[(v2[i]["split"], i.split("_")[0])].append(i)
    keys = sorted(strata)
    per = {k: max(1, round(n * len(strata[k]) / len(both))) for k in keys}
    out = []
    for k in keys:
        out += sorted(strata[k], key=lambda i: h("report", i))[:per[k]]
    return sorted(out, key=lambda i: h("order", i))[:n]


def qa_table(qa1, qa2):
    if not (qa1 and qa2):
        return ""
    a = {r[0]: r for r in qa1["results"]}
    b = {r[0]: r for r in qa2["results"]}
    rows = []
    for name in [r[0] for r in qa2["results"]] + [k for k in a if k not in b]:
        ra, rb = a.get(name), b.get(name)
        tgt = (rb or ra)[1]
        cell = lambda r: f'<td class="{r[3].lower()}">{html.escape(r[2])}</td>' if r else "<td>—</td>"
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{html.escape(tgt)}</td>{cell(ra)}{cell(rb)}</tr>")
    return ("<h2>Acceptance checks (qa.py)</h2><div class=tw><table class=qa><tr><th>check</th><th>target</th><th>v1</th>"
            "<th>v2</th></tr>" + "".join(rows) + "</table></div>")


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e0;--card:#fff;--pass:#1a7f37;--fail:#c62828;--miss:#8a6d00;--warn:#fff3cd}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#141415;--fg:#ececec;--mut:#9a9aa0;--line:#2c2c2f;--card:#1c1c1e;--pass:#4cc26a;--fail:#ff6b6b;--miss:#e0c050;--warn:#3a3218}}
:root[data-theme=dark]{color-scheme:dark;--bg:#141415;--fg:#ececec;--mut:#9a9aa0;--line:#2c2c2f;--card:#1c1c1e;--pass:#4cc26a;--fail:#ff6b6b;--miss:#e0c050;--warn:#3a3218}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}
h1{font-size:22px;text-wrap:balance}h2{font-size:17px;margin-top:28px}
.summary{max-width:75ch;padding-left:20px}.summary li{margin-bottom:6px}
.tw{overflow-x:auto}
table.qa{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}table.qa td,table.qa th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}
.rot,.opts,.qt{overflow-wrap:anywhere}
td.pass{color:var(--pass)}td.fail{color:var(--fail);font-weight:600}td.missing{color:var(--miss)}
.img{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:14px 0;padding:12px;display:grid;grid-template-columns:224px 1fr 1fr;gap:14px}
.img img{width:224px;border-radius:4px}.id{font-size:12px;color:var(--mut);word-break:break-all}
.col h3{font-size:13px;margin:0 0 6px;color:var(--mut)}
.q{border-top:1px solid var(--line);padding:6px 0}.q.aug{opacity:.6}.qt{font-weight:600}
.meta,.rot{font-size:12px;color:var(--mut)}.opts{font-size:13px}.lab{font-size:13px}
.na{margin-left:10px;color:var(--mut)}.na.hi{color:var(--fail);font-weight:600}
.flags span{background:var(--warn);border-radius:4px;font-size:12px;margin-right:6px;padding:1px 6px}
@media (max-width:760px){.img{grid-template-columns:1fr}.img img{width:100%;max-width:320px}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--qa1")
    ap.add_argument("--qa2")
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--only", help="jsonl with image_id per line: sample only these images (pilot)")
    ap.add_argument("--labelled", nargs="*", default=[], metavar="NAME=FILE",
                    help="teacher label files (with per-rotation readouts), e.g. Qwen3=out/v2/pilot3/lab_q3.jsonl")
    ap.add_argument("--summary", help="text file, one finding per line, shown above the checks")
    ap.add_argument("--title", default="Dataset v2 review")
    args = ap.parse_args()
    v1, v2 = load(args.v1), load(args.v2)
    if args.only:
        keep = {json.loads(l)["image_id"] for l in open(args.only)}
        v2 = {i: r for i, r in v2.items() if i in keep}
    rots = {}  # finalize drops the rotations; put them back from the labeller's file for display
    for spec in args.labelled:
        name, path = spec.split("=", 1)
        for r in map(json.loads, open(path)):
            for q in r["questions"]:
                rots.setdefault((r["image_id"], q["q"]), (name, q["rots"]))
    for i, r in v2.items():
        for q in r["questions"]:
            if q["label_kind"] == "teacher_soft" and (i, q["q"]) in rots:
                q["labeller"], q["rots"] = rots[(i, q["q"])]
    qa1 = json.load(open(args.qa1)) if args.qa1 else None
    qa2 = json.load(open(args.qa2)) if args.qa2 else None
    ids = sample(v1, v2, args.n)
    order = lambda qs: sorted(qs, key=lambda q: (is_aug(q) is not None and bool(is_aug(q)), q["source"], q["q"]))
    cards = []
    for i in ids:
        img = thumb(Path(args.images) / f"{i}.jpg")
        cards.append(
            f'<div class="img"><div>{f"<img src={img!r} alt>" if img else "(no image)"}'
            f'<div class="id">{html.escape(i)} · {v2[i]["split"]}</div></div>'
            f'<div class="col"><h3>v1 — {len(v1[i]["questions"])} questions</h3>{"".join(q_html(q) for q in order(v1[i]["questions"]))}</div>'
            f'<div class="col"><h3>v2 — {len(v2[i]["questions"])} questions</h3>{"".join(q_html(q) for q in order(v2[i]["questions"]))}</div></div>')
    lines = [l.strip() for l in open(args.summary)] if args.summary else []
    summary = ("<h2>Summary</h2><ul class=summary>" + "".join(f"<li>{html.escape(l)}</li>" for l in lines if l) + "</ul>") if lines else ""
    # page content only: the Artifact publisher wraps it in the doctype/head/body skeleton
    page = (f"<title>{html.escape(args.title)}</title><style>{CSS}</style><h1>{html.escape(args.title)}: v1 and v2 side by side</h1>"
            f"<p>{len(ids)} images, seeded stratified sample (split × image source). Faded rows are option-count copies. "
            f"Teacher rows list each rotation's raw readout (label:probability in presented order).</p>"
            f"{summary}{qa_table(qa1, qa2)}<h2>Images</h2>{''.join(cards)}")
    Path(args.out).write_text(page)
    print(f"wrote {args.out}: {len(ids)} images, {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
