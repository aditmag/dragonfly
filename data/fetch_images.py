"""Download the images a manifest (out/images.jsonl or out/images_selected.jsonl) refers to,
into $DF_DATA/images/<image_id>.jpg. See IMAGES.md for the rules this follows.
"""
import argparse
import os
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import OUT, ROOT, read_jsonl, write_jsonl

DF_DATA = Path(os.environ.get("DF_DATA", ROOT))  # images land at $DF_DATA/images/, per IMAGES.md


class RangeFile:
    """A read+seek file-like object over an HTTP URL, backed by Range requests (206). Just
    enough for zipfile to open a remote zip and read one member without downloading the rest."""

    def __init__(self, url):
        self.url = url
        self.pos = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            self.size = int(r.headers["Content-Length"])

    def read(self, n=-1):
        # timeout found the hard way 2026-09-30: without it, one hung request (this host is slow
        # and occasionally unresponsive near end-of-file offsets) holds its zip-pool slot's lock
        # forever, and every other worker eventually piles up waiting on a lock that never frees.
        end = self.size - 1 if n is None or n < 0 else min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req, timeout=30).read()
        self.pos += len(data)
        return data

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True


# koniq's host in particular has proven flaky under our own repeated testing 2026-09-30 (even a
# HEAD request timed out at 30s once) -- retry with backoff rather than fail on one bad request.
RETRYABLE = (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError)


def retry(fn, attempts=4, base_delay=2):
    for attempt in range(attempts):
        try:
            return fn()
        except RETRYABLE as e:
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  retry {attempt + 1}/{attempts} after {type(e).__name__}: {e} (waiting {delay}s)", flush=True)
            time.sleep(delay)


# koniq: measured 2026-09-30 at ~24s per individual Range-fetched member, even from an already-
# warm pool -- not a bug, just how slow this host is for random access into this file. But we need
# ~10,000 of its ~10,073 images (~99%), so a one-time whole-archive download (~767MB, a few
# minutes) and reading members off the local copy beats ~10,000 x 24s of Range requests by orders
# of magnitude. Range-based partial fetch stays right for vizwiz, where we need <1% of its archives.
DOWNLOAD_WHOLE = {"http://datasets.vqa.mmsp-kn.de/archives/koniq10k_512x384.zip"}
_download_lock = threading.Lock()


def _ensure_downloaded(archive):
    local_path = DF_DATA / "_archives" / Path(archive).name
    with _download_lock:
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading whole archive (need most of it): {archive} ...", flush=True)
            t0 = time.monotonic()
            tmp = local_path.with_suffix(".tmp")

            def go():
                with urllib.request.urlopen(archive, timeout=120) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            retry(go, attempts=3, base_delay=5)
            tmp.rename(local_path)
            print(f"  downloaded in {time.monotonic() - t0:.0f}s", flush=True)
    return local_path


ZIP_POOL_SIZE = 6  # a single RangeFile/ZipFile per archive serialised every read to 1x throughput
_zip_pools = {}    # (found the hard way: 10k KonIQ images ground through one lock at ~1/sec).
_zip_pools_lock = threading.Lock()  # A small pool of independent handles gives ZIP_POOL_SIZE-way
_zip_next = defaultdict(int)        # concurrency per archive instead, still bounded (not one per image).


def _open_for_pool(archive):
    if archive in DOWNLOAD_WHOLE:
        return open(_ensure_downloaded(archive), "rb")  # zipfile.ZipFile accepts a real file object
    return RangeFile(archive)


def _zip_pool(archive):
    with _zip_pools_lock:
        if archive not in _zip_pools:
            zfs = [retry(lambda: zipfile.ZipFile(_open_for_pool(archive))) for _ in range(ZIP_POOL_SIZE)]
            names = {Path(n).name: n for n in zfs[0].namelist()}  # same for every handle, compute once
            _zip_pools[archive] = {"zfs": zfs, "locks": [threading.Lock() for _ in zfs], "names": names}
        return _zip_pools[archive]


def fetch_zip_member(archive, member, dest):
    pool = _zip_pool(archive)
    if member not in pool["names"]:
        raise FileNotFoundError(f"{member} not found in {archive}")
    with _zip_pools_lock:  # tiny critical section, just picking which slot's lock to take
        i = _zip_next[archive] % ZIP_POOL_SIZE
        _zip_next[archive] += 1
    with pool["locks"][i]:
        data = retry(lambda: pool["zfs"][i].read(pool["names"][member]))
    dest.write_bytes(data)


def fetch_url(url, dest):
    def go():
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    dest.write_bytes(retry(go))


def fetch_one(row):
    dest = DF_DATA / "images" / f"{row['image_id']}.jpg"
    if dest.exists():
        return row["image_id"], "skipped", None
    try:
        if "archive" in row:
            fetch_zip_member(row["archive"], row["member"], dest)
        else:
            fetch_url(row["url"], dest)
        return row["image_id"], "done", None
    except (urllib.error.URLError, urllib.error.HTTPError, FileNotFoundError, TimeoutError) as e:
        return row["image_id"], "missing", str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(OUT / "images_selected.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)  # network-bound; raise if a host allows more
    args = ap.parse_args()

    (DF_DATA / "images").mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(Path(args.manifest)))
    if args.limit:
        rows = rows[: args.limit]

    # Pre-warm every zip archive's connection pool sequentially, single-threaded, BEFORE handing
    # anything to the thread pool. Found the hard way: same-source rows are clustered together
    # (merge.py groups by source), so with lazy pool creation, most/all worker threads can grab a
    # zip-archive row at once and all pile up waiting on the *pool-creation* lock -- stalling the
    # ~130k easy direct-URL skips behind it too, since a blocked worker can't pick up other work.
    pending_archives = {r["archive"] for r in rows
                         if "archive" in r and not (DF_DATA / "images" / f"{r['image_id']}.jpg").exists()}
    for archive in pending_archives:
        print(f"warming connection pool for {archive} ...", flush=True)
        _zip_pool(archive)

    done = skipped = 0
    missing = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, row) for row in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            image_id, status, reason = fut.result()
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                missing.append({"image_id": image_id, "reason": reason})
            if i % 500 == 0:
                print(f"{i}/{len(rows)} ({done} fetched, {skipped} already on disk, {len(missing)} missing)", flush=True)

    if missing:
        write_jsonl(OUT / "missing.jsonl", missing)
    print(f"\ndone: {done} fetched, {skipped} already on disk, {len(missing)} missing -> {DF_DATA / 'images'}")


if __name__ == "__main__":
    main()
