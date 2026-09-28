"""Entry 8 (lit_incremental_oos.md): LLM-synthesised near-out-of-scope negatives for a user's route set.

Abbas, Azmat, Horesh & Yurochkin, "Out-of-Distribution Detection using Synthetic Data Generation" (COLM 2025,
arXiv 2502.03323): an LLM writes OOD proxies (near-OOD from in-domain demonstrations), which train either a
binary head on frozen features or a (K+1)-way model. Their pairs are mostly cross-dataset; CLINC-style
out-of-scope is harder (near-OOD), and an LLM may have memorised CLINC, so contamination is measured.

Phase 1 (generation, GPU): a local instruct model (default Qwen/Qwen3-8B via transformers, bf16, thinking
disabled) is shown one CLINC domain's route names and 8 example requests, and asked for 25 short requests
that fit the same assistant but that none of the routes handles. 10 domains x --rounds prompts. Lines are
cleaned, deduplicated, exact CLINC texts removed, and near-duplicates of CLINC test out-of-scope messages
(word Jaccard >= 0.8) dropped; 5-gram overlap with the test out-of-scope set is reported. Saved to
--synth-file; an existing file is reused. `--gen-model stub` (and --smoke) uses a template generator that
only exercises the pipeline. If the model cannot be downloaded or loaded, the script records that and falls
back to the stub, so the other arms still run (read `generator` in the JSON before trusting synth numbers).

Phase 2 (probes on the frozen trunk, CLINC150 with its 250 real out-of-scope training messages REMOVED):
  inscope_msp    150-way in-scope probe, 1 - max softmax (label-free)
  maha_pp        Mahalanobis++ at the same depth (label-free; entry 3)
  real_250       151-way probe with the real out-of-scope training messages (reference)
  synth_250 / synth_all      151-way probe whose out-of-scope class is synthetic
  bank_250 / bank_all        ... Banking77 messages (learning idea J's "free outliers")
  synth_bank     synthetic + Banking77
Scores: P(out-of-scope). Metrics on the CLINC test set: AUROC, AUPR, FPR@95, in-scope accuracy, and OOS F1
after Saerens EM from the probe's training prior (temperature fitted on in-scope validation rows only, so
no real out-of-scope label is used by the non-reference arms).
Expected: synthetic >= Banking77 negatives by 0.02-0.05 AUROC and within ~0.02 of the real 250.

Usage:
  .venv/bin/python explore/lit_oos_synth_neg.py --smoke
  .venv/bin/python explore/lit_oos_synth_neg.py --out results/tarski/explore_lit_synth_neg.json   # A100: ~20 min incl. download
"""

from __future__ import annotations

import os
import sys

# the generation phase needs the Hub (model download; the GPU job runner exports HF_HUB_OFFLINE=1, so this
# overrides it); every other lit_oos script runs offline. Datasets stay offline (cached).
if "--smoke" not in sys.argv and "stub" not in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "0"

import argparse
import json
import random
import re
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import (REPO, Gauss, Logger, Probe, class_prior, f1_at, get_trunk, humanize, l2n, load_ds,
                            ood_metrics, pick_device, pooled, saerens_em, seed_all, split_index, zscore)

import numpy as np
import torch

PROMPT = (
    "You are helping test a customer-assistant router. The assistant covers the area \"{domain}\" and handles "
    "exactly these request types:\n{routes}\n\nExamples of requests it handles:\n{examples}\n\n"
    "Write {k} new, realistic user requests (5 to 20 words each) that someone could plausibly send to this "
    "same assistant, on topics near \"{domain}\" or everyday life, but that NONE of the request types above "
    "would handle. Vary the wording and the topics. Output one request per line, with no numbering, quotes "
    "or explanations.")


def words(t: str):
    return re.findall(r"[a-z0-9']+", t.lower())


def clean_lines(text: str):
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip().strip('"').strip()
        if 3 <= len(ln.split()) <= 30 and not ln.endswith(":"):
            out.append(ln)
    return out


def generate_llm(model_name: str, prompts, max_new: int, bs: int, log):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model.to(dev).eval()
    outs = []
    for s in range(0, len(prompts), bs):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for p in prompts[s:s + bs]]
        enc = tok(chats, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.9, top_p=0.95,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
        log(f"   generated {min(s + bs, len(prompts))}/{len(prompts)} prompts")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outs


def generate_stub(domains_ex, rounds: int, k: int, rng: random.Random):
    """Pipeline test only: template requests built from other domains' vocabulary."""
    verbs = ["can you explain", "tell me about", "i want to know about", "help me understand", "what do you think of"]
    outs = []
    names = list(domains_ex)
    for dom in names:
        for _ in range(rounds):
            other = rng.choice([d for d in names if d != dom])
            vocab = [w for ex in domains_ex[other] for w in words(ex) if len(w) > 3] or ["weather"]
            outs.append((dom, "\n".join(f"{rng.choice(verbs)} {rng.choice(vocab)} and {rng.choice(vocab)} today" for _ in range(k))))
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gen-model", default="Qwen/Qwen3-8B", help="HF id, or 'stub'")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--per-prompt", type=int, default=25)
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--gen-bs", type=int, default=16)
    ap.add_argument("--synth-file", default=os.path.join(REPO, "results", "tarski", "lit_oos_synth_clinc.jsonl"))
    ap.add_argument("--depths", type=int, nargs="*", default=[4, 22])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.gen_model, args.rounds, args.per_prompt, args.depths = "stub", 2, 10, [22]
        args.synth_file = os.path.join(REPO, "results", "tarski", "lit_oos_synth_clinc_smoke.jsonl")
    out = args.out or ("results/tarski/explore_lit_synth_neg_smoke.json" if args.smoke
                       else "results/tarski/explore_lit_synth_neg.json")
    log = Logger(out)
    seed_all(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()
    ds = load_ds("clinc150", args.smoke, args.seed)
    full = load_ds("clinc150", False, args.seed) if args.smoke else ds
    allx, idx = split_index(ds)
    intents = ds.tasks["intent"].labels
    with open(os.path.join(REPO, "tarski", "resources", "clinc_domains.json")) as f:
        domains = json.load(f)
    by_int = {}
    for e in full.train:
        by_int.setdefault(intents[e.y["intent"]], []).append(e.text)
    domains_ex = {d: [t for i in its for t in by_int.get(i, [])[:5]] for d, its in domains.items()}
    res = {"args": vars(args)}

    # ---------------- phase 1: synthetic near-OOS ----------------
    generator = args.gen_model
    if os.path.exists(args.synth_file) and not args.smoke:
        rows = [json.loads(l) for l in open(args.synth_file)]
        generator = rows[0].get("generator", generator) if rows else generator
        log(f"   reusing {len(rows)} synthetic messages from {args.synth_file} (generator {generator})")
    else:
        raw = []
        if args.gen_model != "stub":
            prompts, doms = [], []
            for dname, its in domains.items():
                for r in range(args.rounds):
                    ex = rng.sample(domains_ex[dname], min(8, len(domains_ex[dname])))
                    prompts.append(PROMPT.format(domain=humanize(dname), routes="\n".join(f"- {humanize(i)}" for i in its),
                                                 examples="\n".join(f"- {x}" for x in ex), k=args.per_prompt))
                    doms.append(dname)
            try:
                outs = generate_llm(args.gen_model, prompts, args.max_new, args.gen_bs, log)
                raw = list(zip(doms, outs))
            except Exception as ex:                       # no GPU / no network / no memory
                log(f"   generation with {args.gen_model} failed ({type(ex).__name__}: {ex}); falling back to the stub")
                res["generation_error"] = f"{type(ex).__name__}: {ex}"
                generator = "stub"
        if not raw:
            generator = "stub"
            raw = generate_stub(domains_ex, args.rounds, args.per_prompt, rng)
        clinc_texts = {e.text.strip().lower() for e in full.train + full.val + full.test}
        test_oos = [e.text for e in full.test if e.y["oos"] == 1]
        test_oos_sets = [set(words(t)) for t in test_oos]
        grams = set()
        for t in test_oos:
            w = words(t)
            grams.update(tuple(w[i:i + 5]) for i in range(len(w) - 4))
        seen, rows, n_raw, n_exact, n_near, n_gram = set(), [], 0, 0, 0, 0
        for dname, text in raw:
            for ln in clean_lines(text):
                n_raw += 1
                key = ln.lower()
                if key in seen:
                    continue
                seen.add(key)
                if key in clinc_texts:
                    n_exact += 1
                    continue
                ws = set(words(ln))
                if ws and max(len(ws & s) / len(ws | s) for s in test_oos_sets) >= 0.8:
                    n_near += 1
                    continue
                w = words(ln)
                if any(tuple(w[i:i + 5]) in grams for i in range(len(w) - 4)):
                    n_gram += 1
                rows.append({"text": ln, "domain": dname, "generator": generator})
        os.makedirs(os.path.dirname(args.synth_file), exist_ok=True)
        with open(args.synth_file, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        res["generation"] = {"generator": generator, "raw_lines": n_raw, "kept": len(rows), "dropped_exact_clinc": n_exact,
                             "dropped_near_dup_test_oos": n_near, "kept_sharing_5gram_with_test_oos": n_gram}
        log(f"   synthetic near-OOS ({generator}): {n_raw} lines -> {len(rows)} kept; dropped {n_exact} exact CLINC, "
            f"{n_near} near-duplicates of test OOS; {n_gram} kept share a 5-gram with test OOS")
    res["generator"] = generator
    synth = [r["text"] for r in rows]
    log("   examples: " + " || ".join(synth[:5]))

    # ---------------- phase 2: probes ----------------
    dev = pick_device(args.smoke)
    trunk = get_trunk(args.smoke)
    oos_i = intents.index("oos")
    in_ids = [i for i in range(len(intents)) if i != oos_i]
    imap = {i: j for j, i in enumerate(in_ids)}
    y_oos = np.array([e.y["oos"] for e in allx])
    y150 = np.array([imap.get(e.y["intent"], -1) for e in allx])
    tr, va, te = idx["train"], idx["val"], idx["test"]
    tr_in, va_in = tr[y_oos[tr] == 0], va[y_oos[va] == 0]
    tr_real_oos = tr[y_oos[tr] == 1]
    bank = load_ds("banking77", args.smoke, args.seed)
    bank_texts = [e.text for e in bank.train]
    F_ = pooled(trunk, [e.text for e in allx], args.depths, ds.max_len, log=log)
    Fs = pooled(trunk, synth, args.depths, ds.max_len, log=log)
    Fb = pooled(trunk, bank_texts, args.depths, ds.max_len, log=log)
    n_real = len(tr_real_oos)
    r_ = np.random.default_rng(args.seed)
    pick = lambda n_all, n: r_.choice(n_all, min(n, n_all), replace=False)
    res["depths"] = {}
    for d in args.depths:
        mu, sd = F_[d][tr_in].mean(0, keepdim=True), F_[d][tr_in].std(0, keepdim=True).clamp_min(1e-4)
        Z, Zs, Zb = (F_[d] - mu) / sd, (Fs[d] - mu) / sd, (Fb[d] - mu) / sd
        yte = y_oos[te]
        arms = {}
        pr = Probe(Z, y150, tr_in, va_in, 150, dev, steps=args.steps, seed=args.seed)
        s = 1 - pr.probs(Z[te]).max(-1)
        arms["inscope_msp"] = {**ood_metrics(s, yte), "inscope_acc": float((pr.probs(Z[te]).argmax(-1) == y150[te])[yte == 0].mean())}
        cen = F_[d][tr_in].mean(0, keepdim=True)
        g = Gauss(l2n(F_[d][tr_in] - cen), y150[tr_in], 150, 0.1)
        arms["maha_pp"] = ood_metrics(g.min_dist(l2n(F_[d][te] - cen)), yte)
        negs = {"real_250": Z[tr_real_oos],
                "synth_250": Zs[pick(len(Zs), n_real)], "synth_all": Zs,
                "bank_250": Zb[pick(len(Zb), n_real)], "bank_all": Zb[pick(len(Zb), max(len(Zs), n_real))],
                "synth_bank": torch.cat([Zs, Zb[pick(len(Zb), len(Zs))]])}
        for name, N in negs.items():
            if len(N) == 0:
                continue
            Xtr = torch.cat([Z[tr_in], N])
            ytr = np.r_[y150[tr_in], np.full(len(N), 150)]
            Xall = torch.cat([Xtr, Z[va_in]])
            yall = np.r_[ytr, y150[va_in]]
            tr_rows = np.arange(len(Xtr))
            va_rows = np.arange(len(Xtr), len(Xall))
            p = Probe(Xall, yall, tr_rows, va_rows, 151, dev, steps=args.steps, seed=args.seed)
            Pt = p.probs(Z[te])
            m = ood_metrics(Pt[:, 150], yte)
            adj, pt = saerens_em(Pt, class_prior(ytr, 151))
            m.update({"oos_f1_em": f1_at((adj.argmax(-1) == 150).astype(int), yte)["f1"],
                      "est_oos_share": float(pt[150]), "n_neg": int(len(N)),
                      "inscope_acc": float((Pt.argmax(-1) == y150[te])[yte == 0].mean())})
            arms[name] = m
        res["depths"][d] = arms
        log(f"-- depth {d}: " + " | ".join(
            f"{k} AUROC {v['auroc']:.3f}" + (f" F1(EM) {v['oos_f1_em']:.3f}" if "oos_f1_em" in v else "") for k, v in arms.items()))
        log.dump(res)
    res["wall_s"] = round(time.time() - t0, 1)
    log.dump(res)
    log(f"done in {res['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()
