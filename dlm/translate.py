"""Translate JevBench states for the census language control, on the CPU (opus-mt), so the same
canvases can be routed in another language.

  .venv/bin/python dlm/translate.py --lang es --tiers original easy --out results/dlm/translated_es.json

Writes {item id: translated state text}. Only the state is translated; the questions (system
prompt) stay in English, so the control isolates the state's language. Long states are split into
paragraphs and sentences to stay within opus-mt's 512-token window.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402

MODELS = {"es": "Helsinki-NLP/opus-mt-en-es", "de": "Helsinki-NLP/opus-mt-en-de", "fr": "Helsinki-NLP/opus-mt-en-fr",
          "zh": "Helsinki-NLP/opus-mt-en-zh"}


def chunks(text: str, max_chars: int = 600):
    """Paragraphs, then sentences, grouped up to max_chars; blank lines are preserved as separators."""
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append(("", False))
            continue
        sents = re.split(r"(?<=[.!?])\s+", para.strip())
        buf = ""
        for s in sents:
            if buf and len(buf) + len(s) + 1 > max_chars:
                out.append((buf, True))
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            out.append((buf, True))
        out.append(("\n", False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(MODELS), required=True)
    ap.add_argument("--tiers", nargs="+", default=["original", "easy"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    torch.set_num_threads(a.threads)
    tok = MarianTokenizer.from_pretrained(MODELS[a.lang])
    model = MarianMTModel.from_pretrained(MODELS[a.lang]).eval()
    recs = data.jevbench_public(tuple(a.tiers))[: a.limit]
    done = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = json.load(f)
    t0 = time.time()
    for n, rec in enumerate(recs, 1):
        if rec["id"] in done:
            continue
        pieces = chunks(rec["state"])
        texts = [t for t, translate in pieces if translate]
        translated = []
        for s in range(0, len(texts), 8):
            batch = tok(texts[s:s + 8], return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                gen = model.generate(**batch, num_beams=2, max_new_tokens=512)
            translated += tok.batch_decode(gen, skip_special_tokens=True)
        it = iter(translated)
        out = "".join((next(it) + " ") if translate else t for t, translate in pieces)
        done[rec["id"]] = re.sub(r" +\n", "\n", out).strip()
        with open(a.out, "w") as f:
            json.dump(done, f, ensure_ascii=False, indent=0)
        if n % 10 == 0 or n == len(recs):
            print(f"  {n}/{len(recs)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)
    print(f"wrote {a.out} ({len(done)} states)")


if __name__ == "__main__":
    main()
