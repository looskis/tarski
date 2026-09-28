"""H3 on a cold disk: what does switching to a task cost when its weights are not in the page cache?

For each artifact (probe branch, 1- and 2-layer branches, a full fine-tuned copy of the base, and laya's
checkpoint as a real decision model) this drops the OS page cache (Linux, needs passwordless sudo), then
times load + move to the device + the first decision on a message, which forces every weight to be read.
It also measures device memory for keeping 5 tasks resident as branches on one base vs as 5 full copies.

Usage (on a Linux GPU box): python -m experiments.switch_cold --out results/tarski/switch_cold.json
"""

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time

import torch

from tarski.branches import BlockBranch, Branch, ProbeBranch
from tarski.trunk import Trunk

MSG = "hey can someone on billing look at the double charge for acme, they're pretty upset"


def drop_caches() -> bool:
    r = subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], capture_output=True)
    return r.returncode == 0


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trunk = Trunk(device=a.device)
    dev = trunk.device
    b = trunk.tokenize([MSG])
    labels = [f"l{i}" for i in range(20)]
    res = {"device": str(dev), "cold_cache": None, "load_first_decision_ms": {}, "bytes": {}, "resident_5_tasks_MB": {}}

    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as d:
        fp = trunk.fingerprint()
        arts = {}
        for kind, br in (("probe", ProbeBranch(11, labels, trunk.hidden)),
                         ("blocks+1", BlockBranch(11, labels, trunk.hidden, 1, trunk)),
                         ("blocks+2", BlockBranch(11, labels, trunk.hidden, 2, trunk))):
            p = os.path.join(d, kind)
            br.save(p, fp, trunk.base)
            arts[kind] = p
            res["bytes"][kind] = os.path.getsize(os.path.join(p, "branch.safetensors"))
        from transformers import AutoModel
        full = os.path.join(d, "full")
        AutoModel.from_pretrained(trunk.base).save_pretrained(full)
        res["bytes"]["full_copy"] = sum(os.path.getsize(os.path.join(full, f)) for f in os.listdir(full))

        def load_branch(kind):
            def f():
                br = Branch.load(arts[kind], trunk)
                taps, ctx = trunk.taps(b["input_ids"], b["attention_mask"], [br.split])
                br.probs(taps[br.split], ctx).cpu()
            return f

        def load_full():
            m = AutoModel.from_pretrained(full).to(dev).eval()
            with torch.no_grad():
                m(**b).last_hidden_state.mean().item()

        def load_laya():
            import laya
            ag = laya.load("convaiinnovations/laya", device=str(dev))
            ag.predict(MSG, {"q": {"type": "noul", "instructions": "Is this about billing?"}})

        cases = {**{k: load_branch(k) for k in arts}, "full_copy": load_full, "laya_checkpoint": load_laya}
        for name, fn in cases.items():
            for mode in ("cold", "warm"):
                ts = []
                for _ in range(a.reps):
                    if mode == "cold":
                        res["cold_cache"] = drop_caches()
                    sync(dev)
                    t = time.perf_counter()
                    with torch.no_grad():
                        fn()
                    sync(dev)
                    ts.append((time.perf_counter() - t) * 1000)
                res["load_first_decision_ms"][f"{name}|{mode}"] = round(statistics.median(ts), 1)
                print(name, mode, res["load_first_decision_ms"][f"{name}|{mode}"], "ms", flush=True)

        if dev.type == "cuda":
            torch.cuda.empty_cache()
            base_mb = torch.cuda.memory_allocated() / 2 ** 20
            brs = [Branch.load(arts["blocks+2"], trunk) for _ in range(5)]
            res["resident_5_tasks_MB"]["base + 5 blocks+2 branches"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
            del brs
            torch.cuda.empty_cache()
            fulls = [AutoModel.from_pretrained(full).to(dev) for _ in range(5)]
            res["resident_5_tasks_MB"]["5 full copies (plus the base already resident)"] = \
                round(torch.cuda.memory_allocated() / 2 ** 20 - base_mb, 1)
            res["resident_5_tasks_MB"]["base alone"] = round(base_mb, 1)
            del fulls
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
