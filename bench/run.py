"""Build the extended eval set and score it with every candidate detector.

    .venv/bin/python bench/run.py          # writes bench/results/scores.csv

Eval images (all normalized identically, cached under bench/data/):
  pilot/A0..A8   aigc-detect's n=50 pilot (25 DIV2K / 25 FLUX-schnell), 9 degradation arms
  new/div2k      the 75 DIV2K-valid photos the pilot didn't use
  new/flux       74 fresh FLUX.1-schnell images (image-gen Modal endpoint)
  new/sdxl       74 SDXL-base images, a different generator family
"""
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import modal

HERE = Path(__file__).parent
DATA, RESULTS = HERE / "data", HERE / "results"
PILOT = Path("/root/projects/aigc-detect/corpus")
sys.path.insert(0, "/root/projects/aigc-detect")
from harness.normalize import normalize_file  # noqa: E402

sys.path.insert(0, str(HERE))
from modal_bench import Scorer, app, div2k_holdout, gen_sdxl  # noqa: E402


def env(path, key):
    for line in Path(path).read_text().splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()


def prompts():
    return [l.split("\t", 1)[1] for l in (HERE / "prompts.tsv").read_text().splitlines() if l]


def gen_flux(out: Path):
    url = env("/root/projects/image-gen/.env", "MODAL_URL")
    key = env("/root/projects/image-gen/.env", "MODAL_API_KEY")
    out.mkdir(parents=True, exist_ok=True)

    def one(i_p):
        i, p = i_p
        dst = out / f"nf_{i:03d}.jpg"
        if dst.exists():
            return
        for attempt in range(3):
            try:
                r = httpx.post(url, params={"prompt": p, "api_key": key}, timeout=300)
                r.raise_for_status()
                raw = out / f"nf_{i:03d}.png"
                raw.write_bytes(r.content)
                normalize_file(raw, dst)
                raw.unlink()
                return
            except httpx.HTTPError as e:
                print(f"flux {i} attempt {attempt}: {e}", flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, enumerate(prompts())))


def main():
    used = [r["orig_file"].split("_")[-1].removesuffix(".png")
            for r in csv.DictReader(open(PILOT / "manifest.csv")) if r["class"] == "real"]
    sdxl_dir, div_dir = DATA / "sdxl", DATA / "div2k"
    with app.run():
        jobs = []
        if not sdxl_dir.exists():
            jobs.append(("sdxl", gen_sdxl.spawn(prompts())))
        if not div_dir.exists():
            jobs.append(("div2k", div2k_holdout.spawn(used)))
        gen_flux(DATA / "flux")
        for name, call in jobs:
            res = call.get()
            d = DATA / name
            d.mkdir(parents=True)
            items = res.items() if isinstance(res, dict) else (
                (f"ns_{i:03d}", b) for i, b in enumerate(res))
            for k, b in items:
                (d / f"{k}.jpg").write_bytes(b)
            print(f"{name}: {len(list(d.iterdir()))} images", flush=True)

        items, labels = [], {}
        for arm in sorted((PILOT / "arms").iterdir()):
            for f in sorted(arm.glob("*.jpg")):
                k = f"pilot/{arm.name}/{f.stem}"
                items.append((k, f.read_bytes()))
                labels[k] = int(f.stem.startswith("fake"))
        for name, label in (("div2k", 0), ("flux", 1), ("sdxl", 1)):
            for f in sorted((DATA / name).glob("*.jpg")):
                k = f"new/{name}/{f.stem}"
                items.append((k, f.read_bytes()))
                labels[k] = label
        print(f"scoring {len(items)} images", flush=True)
        batches = [items[i:i + 64] for i in range(0, len(items), 64)]
        rows = [r for batch in Scorer().score.map(batches) for r in batch]

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "arm", "image", "label", "swinv2", "commfor", "siglip"])
        for r in rows:
            parts = r["key"].split("/")
            w.writerow([parts[0], parts[1], parts[2], labels[r["key"]],
                        f"{r['swinv2']:.6f}", f"{r['commfor']:.6f}", f"{r['siglip']:.6f}"])
    print(f"wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
