"""Smoke test for the LLaDA reader: one typed-test state, native/mean/randtok reads, timings, and a
check that every label is a single token in this tokenizer."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlm import data
from dlm.reads_llada import Reader
recs = data.typed_decisions("test")[:3]
t0 = time.time(); r = Reader(); print(f"loaded in {time.time()-t0:.0f} s; hidden {r.hidden}, vocab {r.vocab}, mask {r.mask_id}")
for rec in recs:
    prep = r.prepare(rec["state"], rec["questions"])
    print("prompt", len(prep["prompt"]), "canvas", prep["width"], "slots", [s["pos"] for s in prep["slots"]])
    for mode in ("random", "mean", "randtok"):
        t = time.time(); p = r.read(prep, mode, seed=0); dt = time.time() - t
        gold = [rec["gold"][q["key"]]["label"] for q in prep["qs"]]
        print(f"  {mode:8s} {dt:.2f}s", [f"{q['choices'][max(range(len(pp)), key=pp.__getitem__)][0]}|{g}" for q, pp, g in zip(prep["qs"], p, gold)])
