"""Inference server for the trained model: one image, any number of typed questions, one forward pass.
Stdlib HTTP only. Listens on localhost by default; open http://127.0.0.1:8800 for the demo page.

    python model/serve.py --run runs/full4b

Endpoints:
  GET  /                -> the page (web/index.html)
  GET  /api/status      -> model, device, temperatures
  POST /api/image       {"image": base64 bytes} -> {"key", "width", "height", "image_tokens", "encode_ms"}
  POST /api/answer      {"key", "questions": [{type, q, options? | scale?, anchors?}]} -> per question
                        {"answers": [[label, p], ...], "na": p, ...} + {"answer_ms"}
Images: the browser resizes before upload; the server caps pixels again (cache.SIZE, 448^2) and keeps the
frozen-ViT output for the last CACHE images, so further questions on the same image skip the vision encoder.

Exposing it (`--host 0.0.0.0`, or behind an HTTPS reverse proxy): add `--public` for per-IP rate limits, a
2 MB request cap and generic error messages, `--origin https://your.site` to let a page on another origin
call it, and `--trust-proxy` to take the client IP from the proxy's X-Forwarded-For.
"""
import argparse
import base64
import hashlib
import io
import json
import sys
import time
from collections import OrderedDict, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import torch

from cache import encode
from heads import Heads
from pack import answers
from train import forward

MAX_BODY = 20 * 1024 * 1024
MAX_BODY_PUBLIC = 2 * 1024 * 1024  # the page sends a <=448^2-pixel JPEG, ~50-150 KB
MAX_QUESTIONS = 20
CACHE = 128  # images whose ViT output is kept (~1-5 MB each on the GPU)
RATE = {"/api/image": (30, 60), "/api/answer": (240, 60)}  # per client IP: requests per window (s); live typing ~2/s
STATE = {}
LOCK = Lock()  # one forward at a time on the GPU
HITS, HITS_LOCK = defaultdict(deque), Lock()


def limited(ip, path):
    """Sliding-window rate limit per (client IP, endpoint)."""
    # ponytail: in-memory and never pruned of idle IPs; fine for a demo, rate-limit in the proxy if it grows
    n, window = RATE[path]
    now = time.monotonic()
    with HITS_LOCK:
        q = HITS[(ip, path)]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= n:
            return True
        q.append(now)
    return False


def validate(qs):
    """Questions from the page -> records pack() accepts, or ValueError with a message for the user."""
    if not isinstance(qs, list) or not 1 <= len(qs) <= MAX_QUESTIONS:
        raise ValueError(f"send 1-{MAX_QUESTIONS} questions")
    out = []
    for i, q in enumerate(qs, 1):
        t, text = q.get("type"), str(q.get("q", "")).strip()
        if t not in ("bool", "choice", "score"):
            raise ValueError(f"question {i}: type must be bool, choice or score")
        if not 1 <= len(text) <= 300:
            raise ValueError(f"question {i}: write a question (up to 300 characters)")
        rec = {"type": t, "q": text}
        if t == "choice":
            opts = [str(o).strip() for o in q.get("options", []) if str(o).strip()]
            if not 2 <= len(opts) <= 8 or len(set(o.lower() for o in opts)) != len(opts) or max(map(len, opts)) > 60:
                raise ValueError(f"question {i}: give 2-8 different options (up to 60 characters each)")
            rec["options"] = opts
        if t == "score":
            lo, hi = (int(x) for x in q.get("scale", [1, 5]))
            if not (0 <= lo < hi <= 10):
                raise ValueError(f"question {i}: the scale must run from 0 or more to at most 10")
            rec["scale"] = [lo, hi]
            anchors = [str(a).strip()[:40] for a in q.get("anchors", []) if str(a).strip()]
            rec["anchors"] = anchors if len(anchors) == 2 else [str(lo), str(hi)]
        out.append(rec)
    return out


def load(args):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    dev, dtype = "cuda", getattr(torch, args.dtype)  # float16 on GPUs without bf16 (checked equal to bf16)
    run = Path(args.run)
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    model = PeftModel.from_pretrained(model, run / "lora").merge_and_unload().eval()  # LoRA folded in: plain forward
    tok = proc.tokenizer
    heads = Heads(model.config.text_config.hidden_size, tok.convert_tokens_to_ids("Yes"),
                  tok.convert_tokens_to_ids("No")).to(dev)
    heads.load_state_dict(torch.load(run / "heads.pt", map_location=dev))
    heads.eval()
    temps = {t: v["T"] for t, v in json.load(open(run / "log.json"))["temperatures"].items()}
    STATE.update(model=model, proc=proc, tok=tok, heads=heads, temps=temps, dev=dev, dtype=dtype,
                 cache=OrderedDict(), run=str(run), name=args.model)


def _jpeg(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


@torch.no_grad()
def image(b64):
    from PIL import Image, ImageOps, UnidentifiedImageError
    Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard: refuse anything larger
    raw = base64.b64decode(b64, validate=True)
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise ValueError("that file isn't an image we can read")
    key = hashlib.sha1(raw).hexdigest()[:16]
    t0 = time.time()
    with LOCK:
        if key not in STATE["cache"]:
            STATE["cache"][key] = encode(STATE["model"], STATE["proc"].image_processor, [img])[0]
            while len(STATE["cache"]) > CACHE:
                STATE["cache"].popitem(last=False)
        STATE["cache"].move_to_end(key)
        entry = STATE["cache"][key]
    torch.cuda.synchronize()
    return {"key": key, "width": img.width, "height": img.height, "image_tokens": entry["embeds"].shape[0],
            "encode_ms": round((time.time() - t0) * 1000, 1)}


@torch.no_grad()
def answer(key, qs):
    entry = STATE["cache"].get(key)
    if entry is None:
        raise ValueError("that image isn't loaded any more; upload it again")
    qs = validate(qs)
    s = STATE
    t0 = time.time()
    with LOCK:
        outs = forward(s["model"], s["heads"], s["tok"], s["model"].config.image_token_id, qs, entry, s["dev"], s["dtype"])
        torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000
    res = []
    for q, o in zip(qs, outs):
        T = s["temps"].get(q["type"], 1.0)
        if q["type"] == "bool":
            p = torch.sigmoid(o["bool"].float() / T).item()
            dist = [("Yes", p), ("No", 1 - p)]
        else:
            dist = list(zip(answers(q), (o["opt"].float() / T).softmax(0).tolist()))
        res.append({**q, "answers": [[a, round(p, 4)] for a, p in dist], "na": round(torch.sigmoid(o["na"].float()).item(), 4)})
    return {"results": res, "answer_ms": round(ms, 1), "per_question_ms": round(ms / len(qs), 1)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: no new TCP connection per request

    def log_message(self, fmt, *a):  # no request logging (questions may be personal)
        pass

    def client_ip(self):
        # behind a proxy: it appends the real client to X-Forwarded-For (last entry)
        fwd = self.headers.get("X-Forwarded-For")
        return fwd.split(",")[-1].strip() if fwd and STATE["proxied"] else self.client_address[0]

    def reply(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin")
        if origin in STATE["origins"]:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # the page posts text/plain (no preflight); answer one anyway
        self.send_response(204)
        if self.headers.get("Origin") in STATE["origins"]:
            self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            s = STATE
            return self.reply(200, {"model": s["name"].split("/")[-1], "device": torch.cuda.get_device_name(),
                                    "temperatures": s["temps"], "max_questions": MAX_QUESTIONS})
        if path in ("/", "/index.html"):
            return self.reply(200, (Path(__file__).parent / "web/index.html").read_bytes(), "text/html; charset=utf-8")
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in RATE:
            return self.reply(404, {"error": "not found"})
        if STATE["public"] and limited(self.client_ip(), path):
            return self.reply(429, {"error": "too many requests; wait a minute and try again"})
        n = int(self.headers.get("Content-Length", 0))
        if n > STATE["max_body"]:
            return self.reply(413, {"error": f"request too large ({STATE['max_body'] // 2**20} MB max)"})
        try:
            body = json.loads(self.rfile.read(n))
            if path == "/api/image":
                return self.reply(200, image(body["image"]))
            return self.reply(200, answer(body["key"], body["questions"]))
        except (ValueError, KeyError, TypeError) as e:  # bad input: say what to fix
            self.reply(400, {"error": str(e) or "bad request"})
        except Exception as e:  # anything else: keep serving; details stay in the server log
            print(f"500 on {path}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            self.reply(500, {"error": "the model hit an error; try again" if STATE["public"] else f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--run", required=True, help="training output dir: lora/, heads.pt, log.json")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to listen on all interfaces")
    ap.add_argument("--dtype", default="bfloat16", help="float16 on GPUs without bf16 (e.g. P100)")
    ap.add_argument("--public", action="store_true", help="per-IP rate limits, 2 MB cap, generic errors")
    ap.add_argument("--origin", action="append", default=[], help="allowed CORS origin, e.g. https://example.com")
    ap.add_argument("--trust-proxy", action="store_true", help="client IP from X-Forwarded-For (behind a proxy)")
    args = ap.parse_args()
    STATE.update(public=args.public, origins=set(args.origin), proxied=args.trust_proxy,
                 max_body=MAX_BODY_PUBLIC if args.public else MAX_BODY)
    load(args)
    # warm-up: the first CUDA forward pays for kernel selection; do it now, not on the user's first question
    from PIL import Image
    key = image(base64.b64encode(_jpeg(Image.new("RGB", (448, 336), (120, 140, 160)))).decode())["key"]
    for _ in range(2):
        answer(key, [{"type": "bool", "q": "Is it blue?"}, {"type": "choice", "q": "Which colour?", "options": ["red", "blue"]},
                     {"type": "score", "q": "How bright?", "scale": [1, 5]}])
    STATE["cache"].clear()
    print(f"serving on http://{args.host}:{args.port}{' (public)' if args.public else ''}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
