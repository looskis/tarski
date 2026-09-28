"""Idea 11: layer-granular continuous batching for requests that need different trunk depths.

Requests arrive (Poisson) needing the trunk to different depths (their deepest requested split). Three
schedulers run the real trunk layers on the device; a virtual clock advances by each executed step's
measured duration, so latencies include real kernel times and padding, while arrivals are simulated:

  static        when idle, take up to B waiting requests and run the trunk to the batch's max depth;
                everyone completes at the end (the batch waits for its deepest member)
  static+exit   same batches, but each request leaves (completes) at its own depth and the batch shrinks
  layerwise     every request carries a layer pointer; each step runs one layer for the (up to B) requests
                sitting at the layer of the oldest unfinished request; new arrivals join at layer 0
                immediately (Orca-style iteration-level scheduling, one iteration = one trunk layer)

Workloads: short messages (CLINC150 texts) with depths drawn from {6, 11, 16, 22}, and long JSON states
(typed-decisions) with depths from {11, 22}. Loads are fractions of the static scheduler's measured
saturated throughput. Reported: p50/p95/p99 latency, throughput, mean executed batch size.

Tested prediction: layerwise cuts p50/p99 latency for shallow requests at low-to-mid load; static+exit
captures most of the gain; at high load static batching's bigger batches win on throughput.

Usage:
  python explore/systems_layerbatch.py --smoke
  python explore/systems_layerbatch.py --out results/tarski/explore_layerbatch.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explore.systems_common import Log, save, sync, threads

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from tarski import data
from tarski.train import autocast
from tarski.trunk import Trunk


class Req:
    __slots__ = ("i", "arrive", "depth", "ids", "layer", "h", "done")

    def __init__(self, i, arrive, depth, ids):
        self.i, self.arrive, self.depth, self.ids = i, arrive, depth, ids
        self.layer, self.h, self.done = 0, None, None


class Executor:
    """Runs one trunk layer for a group of requests (each at the same layer), padding to the longest."""

    def __init__(self, trunk: Trunk):
        self.trunk = trunk
        self.dev = trunk.device

    @torch.no_grad()
    def step(self, group: List[Req]) -> float:
        tr, dev = self.trunk, self.dev
        t0 = time.perf_counter()
        with autocast(dev):
            lens = torch.tensor([len(r.ids) for r in group], device=dev)
            att = (torch.arange(int(lens.max()), device=dev)[None] < lens[:, None]).long()
            if group[0].layer == 0:
                ids = pad_sequence([torch.tensor(r.ids, device=dev) for r in group], batch_first=True,
                                   padding_value=tr.tok.pad_token_id)
                h = tr.model.embeddings(input_ids=ids)
            else:
                h = pad_sequence([r.h for r in group], batch_first=True)
            ctx = tr.context(h, att)
            layer = tr.model.layers[group[0].layer]
            h = layer(h, attention_mask=ctx.masks[layer.attention_type],
                      position_embeddings=ctx.rope[layer.attention_type])
            for j, r in enumerate(group):
                r.h = h[j, : len(r.ids)]
                r.layer += 1
        sync(dev)
        return time.perf_counter() - t0


def simulate(ex: Executor, reqs: List[Req], policy: str, bmax: int) -> Dict:
    for r in reqs:
        r.layer, r.h, r.done = 0, None, None
    clock, pending, active, nxt, steps, sizes = 0.0, [], [], 0, 0, []
    n = len(reqs)

    def admit(t):
        nonlocal nxt
        while nxt < n and reqs[nxt].arrive <= t:
            pending.append(reqs[nxt])
            nxt += 1

    while any(r.done is None for r in reqs):
        admit(clock)
        if policy in ("static", "static+exit"):
            if not pending:
                clock = max(clock, reqs[nxt].arrive)
                continue
            batch, pending[:] = pending[:bmax], pending[bmax:]
            live = list(batch)
            D = max(r.depth for r in batch)
            for _ in range(D):
                grp = live if policy == "static+exit" else batch
                dt = ex.step(grp)
                clock += dt
                steps += 1
                sizes.append(len(grp))
                if policy == "static+exit":
                    for r in [r for r in live if r.layer >= r.depth]:
                        r.done = clock
                    live = [r for r in live if r.layer < r.depth]
                    if not live:
                        break
            for r in batch:
                if r.done is None:
                    r.done = clock
            for r in batch:
                r.h = None
        else:                                                                # layerwise
            active.extend(pending)
            pending.clear()
            if not active:
                clock = max(clock, reqs[nxt].arrive)
                continue
            oldest = min(active, key=lambda r: r.arrive)
            grp = [r for r in active if r.layer == oldest.layer]
            grp = sorted(grp, key=lambda r: r.arrive)[:bmax]
            dt = ex.step(grp)
            clock += dt
            steps += 1
            sizes.append(len(grp))
            for r in grp:
                if r.layer >= r.depth:
                    r.done = clock
                    r.h = None
            active = [r for r in active if r.done is None]
    lat = np.array([r.done - r.arrive for r in reqs]) * 1000
    shallow = np.array([r.depth == min(x.depth for x in reqs) for r in reqs])
    span = max(r.done for r in reqs) - min(r.arrive for r in reqs)
    return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)), "mean_ms": float(lat.mean()),
            "shallow_p50_ms": float(np.percentile(lat[shallow], 50)),
            "throughput_rps": len(reqs) / span, "steps": steps, "mean_batch": float(np.mean(sizes))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="*", default=["short", "long"])
    ap.add_argument("--loads", type=float, nargs="*", default=[0.3, 0.6, 0.9])
    ap.add_argument("--n-req", type=int, default=400)
    ap.add_argument("--bmax", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    depth_mix = {"short": [6, 11, 16, 22], "long": [11, 22]}
    if a.smoke:
        a.device, a.n_req, a.bmax, a.loads, a.workloads = "cpu", 24, 4, [0.5], ["short"]
        depth_mix = {"short": [1, 2, 4]}
    if not a.smoke and not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    ex = Executor(trunk)
    rng = np.random.default_rng(a.seed)
    res = {"bmax": a.bmax, "n_req": a.n_req, "workloads": {}}
    for wl in a.workloads:
        ds = data.load("clinc150" if wl == "short" else "typed-decisions")
        texts = [e.text for e in ds.test]
        ids_all = trunk.token_ids(texts, 64 if wl == "short" else 512)
        depths = depth_mix[wl]
        # warm up, then measure static saturation throughput (everything arrives at t=0)
        warm = [Req(i, 0.0, max(depths), ids_all[i]) for i in range(a.bmax)]
        simulate(ex, warm, "static", a.bmax)
        sat_reqs = [Req(i, 0.0, int(rng.choice(depths)), ids_all[int(rng.integers(len(ids_all)))])
                    for i in range(min(a.n_req, 8 * a.bmax))]
        cap = simulate(ex, sat_reqs, "static", a.bmax)["throughput_rps"]
        entry = {"depths": depths, "static_saturated_rps": cap, "runs": {}}
        log(f"== workload {wl}: depths {depths}, bmax {a.bmax}, static saturated throughput {cap:.1f} req/s")
        for load in a.loads:
            lam = load * cap
            gaps = rng.exponential(1 / lam, a.n_req)
            arrivals = np.cumsum(gaps) - gaps[0]
            pick = rng.integers(len(ids_all), size=a.n_req)
            ds_ = rng.choice(depths, size=a.n_req)
            for policy in ("static", "static+exit", "layerwise"):
                reqs = [Req(i, float(arrivals[i]), int(ds_[i]), ids_all[int(pick[i])]) for i in range(a.n_req)]
                r = simulate(ex, reqs, policy, a.bmax)
                entry["runs"][f"load={load}|{policy}"] = r
                log(f"   load {load:.1f} {policy:12s} p50 {r['p50_ms']:7.1f} p95 {r['p95_ms']:7.1f} p99 "
                    f"{r['p99_ms']:7.1f} ms | shallow p50 {r['shallow_p50_ms']:7.1f} | {r['throughput_rps']:6.1f} req/s "
                    f"| mean batch {r['mean_batch']:.1f}")
            save(res | {"workloads": {**res["workloads"], wl: entry}}, a.out)
        res["workloads"][wl] = entry
        save(res, a.out)
    log("done")


if __name__ == "__main__":
    main()
