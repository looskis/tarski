"""Day-0 routes from descriptions only: a local LLM writes the training messages (lit_decision_models #6).

A new Slack route starts as a name and a one-line description and no labelled messages. Here a local LLM
(default Qwen/Qwen3-8B via transformers, thinking disabled) writes example messages for every CLINC150 intent
from its name and domain alone, tarski branches train on those in seconds, and we score on the real CLINC test
set. Sources: Curate-Train-Refine (Maheshwari & El Haddad, arXiv 2601.16530, 2026), Incubator (Peng & Shang,
arXiv 2404.10877), Vajjala & Shimangaud (arXiv 2502.11830).

Rounds and arms (each with probe@22 and blocks@11+2, trained with tarski.train.train_branch, >= 300 steps):
  synth         round 1: --n-per-intent generated messages per intent, 150-way
  synth+none    + generated out-of-scope messages as a 151st "none" class
  refine        round 2 (Curate-Train-Refine's loop, zero real labels): the round-1 probe's most confused intent
                pairs on held-out synthetic data (topped up with the most similar class centroids) get
                --n-refine contrastive messages each way ("clearly A, could not be mistaken for B")
  refine+real5  round 2 data plus 5 real labelled messages per intent
  real5         5 real messages per intent only (same shots)
  real_full     all real in-scope training messages (reference)
Metrics on the real test split: in-scope accuracy and macro-F1 (150-way), oos AUROC (max probability, and
1 - p(none) where trained), oos recall/F1 at the threshold keeping 95% of in-scope *synthetic* validation
messages (no real labels used). Generated data is cached next to the output (<out>_data.json).

Usage:
  .venv/bin/python explore/lit_dm_synthroutes.py --smoke
  .venv/bin/python explore/lit_dm_synthroutes.py --out results/tarski/explore_synthroutes.json          # A100
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "0"          # the LLM is a first download on the GPU box
os.environ["HF_DATASETS_OFFLINE"] = "0"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Sequence  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lit_dm_common as C  # noqa: E402
from tarski.train import FeatureCache  # noqa: E402
from tarski.trunk import Trunk  # noqa: E402

OOS_TOPICS = ["science and nature", "history and famous people", "sports", "movies, music and celebrities",
              "health and medicine", "geography and countries", "technology products", "philosophy",
              "animals", "shopping for things the assistant cannot buy", "legal questions", "religion",
              "video games", "fashion", "politics", "space and astronomy", "cars and mechanics (repairs)",
              "cooking techniques", "language and grammar", "math problems"]


# ---------------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------------

class Writer:
    def __init__(self, name: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prompts: Sequence[str], max_new_tokens: int, bs: int = 16, seed: int = 0) -> List[str]:
        torch.manual_seed(seed)
        out = []
        for s in range(0, len(prompts), bs):
            chats = []
            for p in prompts[s:s + bs]:
                kw = {"tokenize": False, "add_generation_prompt": True}
                try:
                    chats.append(self.tok.apply_chat_template([{"role": "user", "content": p}], enable_thinking=False, **kw))
                except TypeError:
                    chats.append(self.tok.apply_chat_template([{"role": "user", "content": p}], **kw))
            enc = self.tok(chats, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8,
                                      top_p=0.95, top_k=40, pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out


def parse_lines(text: str, n: int) -> List[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    seen, out = set(), []
    for line in text.splitlines():
        line = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line).strip().strip('"').strip("'").strip()
        if not (3 <= len(line) <= 200) or line.lower().startswith(("here are", "sure", "intent:")):
            continue
        if line.lower() in seen:
            continue
        seen.add(line.lower())
        out.append(line)
    return out[:n]


def intent_prompt(name: str, domain: str, n: int) -> str:
    return (f"You are creating training data for the intent classifier of a virtual assistant.\n"
            f"Intent: \"{name}\" (domain: {domain}).\n"
            f"Write {n} different messages that a user might send to the assistant with exactly this intent. "
            f"Vary wording, length and tone (casual, terse, polite; the occasional typo). "
            f"Output one message per line, no numbering, nothing else.")


def oos_prompt(domains: Sequence[str], topic: str, n: int) -> str:
    return (f"A virtual assistant ONLY handles these areas: {', '.join(domains)}.\n"
            f"Write {n} different messages a user might send to it that fall outside all of these areas, "
            f"for example about {topic}. Output one message per line, no numbering, nothing else.")


def refine_prompt(a: str, da: str, b: str, db: str, n: int) -> str:
    return (f"Two intents of a virtual assistant are easy to confuse:\nA: \"{a}\" (domain: {da})\n"
            f"B: \"{b}\" (domain: {db})\nWrite {n} messages that clearly have intent A and could not be "
            f"mistaken for intent B. Output one message per line, no numbering, nothing else.")


# ---------------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-per-intent", type=int, default=25)
    ap.add_argument("--n-oos", type=int, default=500, help="generated out-of-scope messages")
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--n-refine", type=int, default=10)
    ap.add_argument("--probe-depth", type=int, default=22)
    ap.add_argument("--blocks", type=int, nargs=2, default=[11, 2], metavar=("SPLIT", "DEPTH"))
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.llm = "Qwen/Qwen3-0.6B"
        args.base = args.base or C.SMOKE_BASE
        args.n_per_intent, args.n_oos, args.n_pairs, args.n_refine, args.shots = 6, 12, 1, 3, 2
        args.probe_depth, args.blocks = 7, [4, 2]
        args.out = args.out or "results/tarski/explore_synthroutes_smoke.json"
    log = C.Log(args.out)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.out.replace(".json", "_data.json")
    gen = C.load_json(data_path)

    ds = C.load_dataset_by_name("clinc150")
    labs = ds.tasks["intent"].labels
    with open(os.path.join(C.REPO, "tarski", "resources", "clinc_domains.json")) as f:
        domains = json.load(f)
    dom_of = {i: d.replace("_", " ") for d, its in domains.items() for i in its}
    inscope = [i for i, l in enumerate(labs) if l != "oos"]
    if args.smoke:
        inscope = inscope[:4]
        ds = C.subsample(ds, 200, 40, 120, classes={"intent": inscope + [labs.index("oos")]})
    names = [C.humanize(labs[i]) for i in inscope]
    doms = [dom_of[labs[i]] for i in inscope]
    K = len(inscope)

    # --- round 1 generation ------------------------------------------------------------------------------
    writer = None

    def get_writer():
        nonlocal writer
        if writer is None:
            t0 = time.time()
            writer = Writer(args.llm, dev)
            log(f"   loaded {args.llm} on {dev} in {time.time() - t0:.0f}s")
        return writer

    if "intents" not in gen:
        t0 = time.time()
        raw = get_writer()([intent_prompt(n, d, args.n_per_intent + 5) for n, d in zip(names, doms)],
                           max_new_tokens=30 * (args.n_per_intent + 5), seed=args.seed)
        gen["intents"] = {labs[i]: parse_lines(r, args.n_per_intent) or [n] for i, r, n in zip(inscope, raw, names)}
        n_oos_prompts = max(1, -(-args.n_oos // 25))
        per = min(25, args.n_oos)
        raw = get_writer()([oos_prompt(sorted({d for d in doms}), OOS_TOPICS[j % len(OOS_TOPICS)], per + 3)
                            for j in range(n_oos_prompts)], max_new_tokens=30 * (per + 3), seed=args.seed + 1)
        gen["oos"] = [l for r in raw for l in parse_lines(r, per)][: args.n_oos]
        gen["round1_seconds"] = round(time.time() - t0, 1)
        C.dump(gen, data_path)
    counts = [len(gen["intents"][labs[i]]) for i in inscope]
    log(f"== round 1: {sum(counts)} intent messages (min {min(counts)}, max {max(counts)} per intent), "
        f"{len(gen['oos'])} out-of-scope ({gen.get('round1_seconds')}s)")

    # --- trunk features ----------------------------------------------------------------------------------
    trunk = C.load_trunk(args.base, args.device)
    rng = np.random.default_rng(args.seed)
    syn_x, syn_y, syn_split = [], [], []
    for c, i in enumerate(inscope):
        msgs = gen["intents"][labs[i]]
        order = rng.permutation(len(msgs))
        for r, j in enumerate(order):
            syn_x.append(msgs[j])
            syn_y.append(c)
            syn_split.append("val" if r < max(1, len(msgs) // 5) else "train")
    for r, m in enumerate(gen["oos"]):
        syn_x.append(m)
        syn_y.append(K)
        syn_split.append("val" if r % 5 == 0 else "train")
    remap = {o: j for j, o in enumerate(inscope)}
    real = ds.train + ds.val + ds.test
    real_y = np.array([remap.get(e.y["intent"], -1) for e in real])
    real_split = np.array(["train"] * len(ds.train) + ["val"] * len(ds.val) + ["test"] * len(ds.test))
    depths = sorted({args.probe_depth, args.blocks[0]})

    def build_cache(extra: List[str]):
        texts = syn_x + extra + [e.text for e in real]
        return FeatureCache(trunk, texts, depths, ds.max_len)

    res = C.load_json(args.out) if not args.smoke else {}
    res.setdefault("config", vars(args) | {"base": trunk.base, "n_intents": K})
    res.setdefault("arms", {})
    res["generation"] = {"per_intent": dict(zip([labs[i] for i in inscope], counts)), "n_oos": len(gen["oos"]),
                         "examples": {labs[i]: gen["intents"][labs[i]][:3] for i in inscope[:5]},
                         "oos_examples": gen["oos"][:5]}

    def evaluate_arm(key, cache, tr, y_tr, va, y_va, with_none: bool, offset: int, select: bool = True):
        te_in = [offset + i for i in range(len(real)) if real_split[i] == "test" and real_y[i] >= 0]
        te_oos = [offset + i for i in range(len(real)) if real_split[i] == "test" and real_y[i] < 0]
        y_te = real_y[[i - offset for i in te_in]]
        n_out = K + 1 if with_none else K
        for kind, sp, dp in (("probe", args.probe_depth, 0), ("blocks", args.blocks[0], args.blocks[1])):
            name = f"{key}|{kind}"
            if name in res["arms"]:
                continue
            br = C.make(kind, sp, dp, [str(j) for j in range(n_out)], trunk)
            info = C.run_branch(br, cache, tr, torch.tensor(y_tr), va, torch.tensor(y_va), kind, seed=args.seed,
                                select=select, log=log)
            P_in, P_oos, P_va = (C.probs_of(br, cache, ids) for ids in (te_in, te_oos, va))
            m = C.metrics(P_in[:, :K] / P_in[:, :K].sum(-1, keepdims=True), y_te)
            m.update({k: info[k] for k in ("opt_steps", "epochs_run", "train_s", "n_train", "selection")})
            va_in = np.array(y_va) < K
            fns = {"msp": lambda p: p[:, :K].max(-1)}
            if with_none:
                fns["1-p_none"] = lambda p: 1 - p[:, K]
            for sname, fn in fns.items():
                thr = float(np.quantile(fn(P_va[va_in]), 0.05)) if va_in.any() else None
                m[f"oos[{sname}]"] = C.oos_metrics(fn(P_in), fn(P_oos), thr)
            res["arms"][name] = m
            best = max((v for k2, v in m.items() if k2.startswith("oos[")), key=lambda v: v.get("auroc", 0))
            log(f"   {name:24s} in-scope acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} | oos auroc "
                f"{best.get('auroc', float('nan')):.3f} F1 {best.get('oos_f1', float('nan')):.3f} ({info['n_train']} train)")
            C.dump(res, args.out)
            del br
            C.gpu_gc()

    def syn_rows(with_none: bool, extra_n: int = 0, extra_y=None):
        tr = [i for i in range(len(syn_x)) if syn_split[i] == "train" and (with_none or syn_y[i] < K)]
        va = [i for i in range(len(syn_x)) if syn_split[i] == "val" and (with_none or syn_y[i] < K)]
        tr += list(range(len(syn_x), len(syn_x) + extra_n))
        y = np.array(syn_y + list(extra_y or []))
        return tr, y[tr], va, y[va]

    # round 1
    cache = build_cache([])
    off = len(syn_x)
    log(f"== trunk {trunk.base}: cached {len(syn_x)} synthetic + {len(real)} real messages")
    tr, ytr, va, yva = syn_rows(False)
    evaluate_arm("synth", cache, tr, ytr, va, yva, with_none=False, offset=off)
    tr, ytr, va, yva = syn_rows(True)
    evaluate_arm("synth+none", cache, tr, ytr, va, yva, with_none=True, offset=off)

    # --- round 2: refine the most confused pairs --------------------------------------------------------------
    if "refine" not in gen:
        pr = C.make("probe", args.probe_depth, 0, [str(j) for j in range(K + 1)], trunk)
        C.run_branch(pr, cache, tr, torch.tensor(ytr), va, torch.tensor(yva), "probe", seed=args.seed)
        pv = C.probs_of(pr, cache, va).argmax(-1)
        conf = {}
        for t_, p_ in zip(yva, pv):
            if t_ < K and p_ < K and t_ != p_:
                conf[(int(t_), int(p_))] = conf.get((int(t_), int(p_)), 0) + 1
        pairs = [tuple(sorted(p)) for p, _ in sorted(conf.items(), key=lambda kv: -kv[1])]
        cent = C.pooled(cache, args.probe_depth, [i for i in range(len(syn_x)) if syn_y[i] < K])
        ys = np.array([y for y in syn_y if y < K])
        cm = torch.stack([cent[ys == c].mean(0) for c in range(K)])
        sim = C.cosine_logits(cm, cm, cm.mean(0)).numpy()
        np.fill_diagonal(sim, -9)
        for flat in np.argsort(-sim, axis=None):
            a, b = divmod(int(flat), K)
            pairs.append(tuple(sorted((a, b))))
        uniq = []
        for p in pairs:
            if p not in uniq:
                uniq.append(p)
        uniq = uniq[: args.n_pairs]
        prompts, meta = [], []
        for a, b in uniq:
            prompts.append(refine_prompt(names[a], doms[a], names[b], doms[b], args.n_refine + 2))
            meta.append(a)
            prompts.append(refine_prompt(names[b], doms[b], names[a], doms[a], args.n_refine + 2))
            meta.append(b)
        t0 = time.time()
        raw = get_writer()(prompts, max_new_tokens=30 * (args.n_refine + 2), seed=args.seed + 2)
        gen["refine"] = {"pairs": [[labs[inscope[a]], labs[inscope[b]]] for a, b in uniq],
                         "from_confusions": len(conf),
                         "messages": [[int(c), m] for c, r in zip(meta, raw) for m in parse_lines(r, args.n_refine)],
                         "seconds": round(time.time() - t0, 1)}
        C.dump(gen, data_path)
        del pr
    if writer is not None:
        del writer.model
        writer = None
        C.gpu_gc()
    ref = gen["refine"]
    res["generation"]["refine_pairs"] = ref["pairs"]
    log(f"== round 2: {len(ref['messages'])} refinement messages for {len(ref['pairs'])} pairs "
        f"({ref['from_confusions']} confused pairs seen on synthetic validation data)")
    extra_x = [m for _, m in ref["messages"]]
    extra_y = [c for c, _ in ref["messages"]]
    cache = build_cache(extra_x)
    off = len(syn_x) + len(extra_x)
    tr, ytr, va, yva = syn_rows(True, len(extra_x), extra_y)
    evaluate_arm("refine", cache, tr, ytr, va, yva, with_none=True, offset=off)

    # --- real labels -----------------------------------------------------------------------------------------
    real_pool = [i for i in range(len(real)) if real_split[i] == "train" and real_y[i] >= 0]
    shots = C.kshot(real_y, real_pool, args.shots, args.seed)
    y_all = np.array(syn_y + extra_y + real_y.tolist())
    tr2 = tr + [off + i for i in shots]
    evaluate_arm(f"refine+real{args.shots}", cache, tr2, y_all[tr2], va, yva, with_none=True, offset=off)
    rv = C.kshot(real_y, [i for i in range(len(real)) if real_split[i] == "val" and real_y[i] >= 0], args.shots, args.seed + 1)
    tr3, va3 = [off + i for i in shots], [off + i for i in rv]
    evaluate_arm(f"real{args.shots}", cache, tr3, y_all[tr3], va3, y_all[va3], with_none=False, offset=off,
                 select=False)            # few-shot protocol, as in lit_dm_kshot: no epoch selection
    tr4 = [off + i for i in real_pool]
    va4 = [off + i for i in range(len(real)) if real_split[i] == "val" and real_y[i] >= 0]
    evaluate_arm("real_full", cache, tr4, y_all[tr4], va4, y_all[va4], with_none=False, offset=off)
    C.dump(res, args.out)
    log("done")


if __name__ == "__main__":
    main()
