"""Download the CDN images of the URL-only datasets into data/raw/<name>-original/<split>/
and point each record's `file` at the local copy (the signed URLs expire).
Run: uv run scripts/download_images.py [name ...]
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import RAW

SOURCES = ["golf-ball-detection", "golf-ball"]


def fetch(args):
    url, dest = args
    if dest.exists():
        return True
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as e:
        print(f"FAILED {dest.name}: {e}")
        return False


def main():
    for name in sys.argv[1:] or SOURCES:
        path = RAW / f"{name}.ndjson"
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        jobs = []
        for rec in lines[1:]:
            rel = Path(f"{name}-original") / rec["split"] / Path(rec["file"]).name
            rec["_dest"] = rel
            jobs.append((rec["url"], RAW / rel))
        with ThreadPoolExecutor(8) as ex:
            ok = list(ex.map(fetch, jobs))
        for rec, good in zip(lines[1:], ok):
            rel = rec.pop("_dest")
            if good:  # keep the old file/url untouched for failures
                rec["file"] = rel.as_posix()
        path.write_text("".join(json.dumps(l, ensure_ascii=False, separators=(",", ":")) + "\n" for l in lines),
                        encoding="utf-8")
        print(f"{name}: downloaded {sum(ok)}/{len(ok)}")


if __name__ == "__main__":
    main()
