"""Drop-in trunks with a zero-shot readout (lit_decision_models #3).

tarski's trunk is answerdotai/ModernBERT-base, whose only label-free readout is its MLM head. Same-architecture
checkpoints load through `Trunk(base=...)` unchanged, and some of them come with a zero-shot readout:

  modernbert  answerdotai/ModernBERT-base (the baseline, same protocol)
  gte         Alibaba-NLP/gte-modernbert-base: contrastive embedding model fine-tuned from ModernBERT-base
              (CLS pooling). Zero-shot route = cosine(message, label text) at the top layer.
  gliclass    the encoder of knowledgator/gliclass-modern-base-v3.0 (Stepanov et al., arXiv 2508.07662), a
              ModernBERT-base fine-tuned for zero-shot classification with labels in the input. Converted to a
              plain ModernBertModel (vocab 50370, two added tokens) and cached under ~/.cache/tarski/lit_dm/.
              Its own readouts are reproduced from the GLiClass source (uni-encoder, "first" pooling,
              text/class projectors, dot-product scorer):
                joint        "<<LABEL>>l1<<LABEL>>l2...<<SEP>>text", all labels in one input (re-reads the
                             message per label set, like laya)
                joint_chunk  the same with at most 25 labels per input (the model's training maximum),
                             argmax across chunks
                cached       message alone and each label alone ("<<LABEL>>name<<SEP>>"): label embeddings can
                             be cached and the message is encoded once, as tarski wants
  ettin       jhu-clsp/ettin-encoder-150m (lit_decision_models #7's encoder arm)

Per base and dataset (Banking77, CLINC150's 151-way intent task, typed-decisions customer_service):
  - probe scan at several depths, blocks branches (the thesis's configurations), all trained with
    tarski.train.train_branch (>= 300 optimiser steps; standard selection);
  - zero-shot: cosine between pooled message and pooled label text (mean and CLS pooling, top layer and mid
    depths, two templates), accuracy and (CLINC) oos AUROC / oos F1 at a threshold chosen on in-scope
    validation data only;
  - for gliclass, its native joint / chunked / cached readouts.

Usage:
  .venv/bin/python explore/lit_dm_trunkswap.py --smoke
  .venv/bin/python explore/lit_dm_trunkswap.py --base gte --out results/tarski/explore_trunkswap_gte.json
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "0"          # first download of new bases on the GPU box
os.environ["HF_DATASETS_OFFLINE"] = "0"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional, Sequence  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lit_dm_common as C  # noqa: E402
from tarski.train import FeatureCache, autocast  # noqa: E402
from tarski.trunk import Trunk  # noqa: E402

ALIASES = {"modernbert": "answerdotai/ModernBERT-base", "gte": "Alibaba-NLP/gte-modernbert-base",
           "ettin": "jhu-clsp/ettin-encoder-150m", "ettin-17m": "jhu-clsp/ettin-encoder-17m"}
GLICLASS_REPO = "knowledgator/gliclass-modern-base-v3.0"
TEMPLATES = {"name": "{name}", "about": "This message is about {name}."}


# ---------------------------------------------------------------------------------------------------
# GLiClass conversion and native readouts
# ---------------------------------------------------------------------------------------------------

def gliclass_encoder(repo: str = GLICLASS_REPO) -> Dict:
    """Export GLiClass's encoder as a plain ModernBertModel directory; return it with the readout weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer, ModernBertModel

    src = snapshot_download(repo)
    with open(os.path.join(src, "config.json")) as f:
        gcfg = json.load(f)
    out_dir = os.path.expanduser(f"~/.cache/tarski/lit_dm/{repo.split('/')[-1]}-encoder")
    sd = load_file(os.path.join(src, "model.safetensors"))
    enc = {k[len("model.encoder_model."):]: v for k, v in sd.items() if k.startswith("model.encoder_model.")}
    if not os.path.exists(os.path.join(out_dir, "model.safetensors")):
        cfg = AutoConfig.from_pretrained(gcfg.get("encoder_model_name", "answerdotai/ModernBERT-base"))
        cfg.vocab_size = int(gcfg["vocab_size"])
        model = ModernBertModel(cfg)
        missing, unexpected = model.load_state_dict(enc, strict=False)
        if unexpected or [m for m in missing if "rotary" not in m]:
            raise RuntimeError(f"GLiClass encoder conversion: missing {missing}, unexpected {unexpected}")
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        AutoTokenizer.from_pretrained(src).save_pretrained(out_dir)
    heads = {k[len("model."):]: v.float() for k, v in sd.items() if not k.startswith("model.encoder_model.")}
    return {"dir": out_dir, "heads": heads, "class_token": int(gcfg["class_token_index"]),
            "sep_token": int(gcfg["text_token_index"]), "max_num_classes": int(gcfg.get("max_num_classes", 25)),
            "act": gcfg.get("projector_hidden_act", "gelu")}


def _project(heads: Dict, name: str, x: torch.Tensor) -> torch.Tensor:
    w1, b1 = heads[f"{name}.linear_1.weight"].to(x.device), heads[f"{name}.linear_1.bias"].to(x.device)
    w2, b2 = heads[f"{name}.linear_2.weight"].to(x.device), heads[f"{name}.linear_2.bias"].to(x.device)
    return F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)


@torch.no_grad()
def gliclass_joint(trunk: Trunk, g: Dict, texts: Sequence[str], labels: Sequence[str], chunk: Optional[int],
                   bs: int = 16, max_len: int = 2048) -> np.ndarray:
    """Native uni-encoder scores [n, n_labels]; with `chunk`, labels are split into groups of that size."""
    groups = [list(range(len(labels)))] if not chunk else \
        [list(range(s, min(s + chunk, len(labels)))) for s in range(0, len(labels), chunk)]
    out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    for grp in groups:
        prefix = "".join(f"<<LABEL>>{labels[i]}" for i in grp) + "<<SEP>>"
        for s in range(0, len(texts), bs):
            enc = trunk.tokenize([prefix + t for t in texts[s:s + bs]], max_len=max_len)
            with autocast(trunk.device):
                h = trunk.model(**enc).last_hidden_state.float()
            txt = _project(g["heads"], "text_projector", h[:, 0])
            pos = enc["input_ids"] == g["class_token"]
            cls = torch.stack([h[b][pos[b]][: len(grp)] for b in range(h.shape[0])])     # [b, |grp|, d]
            cls = _project(g["heads"], "classes_projector", cls)
            out[s:s + len(txt), grp] = torch.einsum("bd,bkd->bk", txt, cls).cpu().numpy()
    return out


@torch.no_grad()
def gliclass_cached(trunk: Trunk, g: Dict, texts: Sequence[str], labels: Sequence[str], bs: int = 64) -> np.ndarray:
    lab = []
    for s in range(0, len(labels), bs):
        enc = trunk.tokenize([f"<<LABEL>>{l}<<SEP>>" for l in labels[s:s + bs]])
        with autocast(trunk.device):
            h = trunk.model(**enc).last_hidden_state.float()
        pos = (enc["input_ids"] == g["class_token"]).float().argmax(-1)
        lab.append(_project(g["heads"], "classes_projector", h[torch.arange(len(h)), pos]))
    lab = torch.cat(lab)
    out = []
    for s in range(0, len(texts), bs):
        enc = trunk.tokenize(list(texts[s:s + bs]))
        with autocast(trunk.device):
            h = trunk.model(**enc).last_hidden_state.float()
        out.append((_project(g["heads"], "text_projector", h[:, 0]) @ lab.T).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------------

def zero_shot_eval(scores_te: np.ndarray, y_te: np.ndarray, scores_va: Optional[np.ndarray], oos_mask_te: np.ndarray,
                   oos_mask_va: Optional[np.ndarray]) -> Dict:
    """scores over in-scope labels only. Accuracy on in-scope test rows; with oos rows, AUROC of the max score
    and oos recall/F1 at the threshold that keeps 95% of in-scope validation rows."""
    ins = ~oos_mask_te
    out = {"acc": float((scores_te[ins].argmax(-1) == y_te[ins]).mean())}
    if oos_mask_te.any():
        thr = None
        if scores_va is not None and oos_mask_va is not None and (~oos_mask_va).any():
            thr = float(np.quantile(scores_va[~oos_mask_va].max(-1), 0.05))
        out["oos"] = C.oos_metrics(scores_te[ins].max(-1), scores_te[oos_mask_te].max(-1), thr)
    return out


def run_dataset(name: str, trunk: Trunk, args, log, res: Dict, g: Optional[Dict]) -> None:
    ds = C.load_dataset_by_name(name)
    if name == "clinc150":                           # the 151-way intent decision (oos is one of its labels)
        ds = C.restrict_tasks(ds, ["intent"], "clinc150")
    if args.smoke:
        if name == "typed-cs":
            ds = C.restrict_tasks(ds, list(ds.tasks)[:2], "typed-cs")
            ds = C.subsample(ds, 60, 40, 40)
            ds.max_len = 128
        else:
            labs = ds.tasks["intent"].labels
            keep = [i for i, l in enumerate(labs) if l != "oos"][:6] + ([labs.index("oos")] if "oos" in labs else [])
            ds = C.subsample(ds, 240, 60, 80, classes={"intent": keep})
    n = trunk.n_layers
    scan = [d for d in (args.probe_depths or ([6, 14, n] if name == "typed-cs" else [4, 8, 11, 14, 18, n])) if d <= n]
    if args.smoke:
        scan = sorted({max(1, n // 3), n // 2, n})
    blocks = {"banking77": [(14, 2), (18, 1)], "clinc150": [(11, 2)], "typed-cs": [(14, 2)]}[name]
    if args.smoke or n < 22:
        blocks = [(max(1, n // 2), 2)]
    blocks = [(k, d) for k, d in blocks if k + d <= n]
    depths = sorted(set(scan) | {k for k, _ in blocks})
    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    split_of = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * len(ds.test))
    cache = FeatureCache(trunk, [e.text for e in allx], depths, ds.max_len)
    log(f"== {ds.summary()} | cached {depths} in {cache.seconds:.1f}s ({cache.bytes() / 1e6:.0f} MB)")
    out = res["datasets"].setdefault(name, {})

    # --- trained branches -------------------------------------------------------------------------
    def train_eval(kind, split, depth):
        per = {}
        for t in ds.tasks:
            ids = {s: [i for i in range(len(allx)) if split_of[i] == s and t in allx[i].y] for s in ("train", "val", "test")}
            y = {s: np.array([allx[i].y[t] for i in ids[s]]) for s in ids}
            soft = None
            if all(t in allx[i].soft for i in ids["train"]):
                soft = torch.tensor(np.stack([allx[i].soft[t] for i in ids["train"]]))
            br = C.make(kind, split, depth, ds.tasks[t].labels, trunk)
            info = C.run_branch(br, cache, ids["train"], torch.tensor(y["train"]), ids["val"], torch.tensor(y["val"]),
                                kind, soft_tr=soft, log=log)
            p = C.probs_of(br, cache, ids["test"])
            m = C.metrics(p, y["test"])
            m.update({k: info[k] for k in ("opt_steps", "epochs_run", "train_s", "temperature")})
            labs = ds.tasks[t].labels
            if "oos" in labs and t == "intent":
                o = labs.index("oos")
                ins = y["test"] != o
                m["oos_via_p_oos"] = C.oos_metrics(1 - p[ins, o], 1 - p[~ins, o])
            per[t] = m
            del br
            C.gpu_gc()
        mean = {k: float(np.mean([v[k] for v in per.values()])) for k in ("acc", "macro_f1", "ece")}
        return {"mean": mean, "tasks": per}

    for d in scan:
        key = f"probe@{d}"
        if key in out.get("branches", {}):
            continue
        log(f"-- {name} {key}")
        out.setdefault("branches", {})[key] = train_eval("probe", d, 0)
        log(f"   {key}: mean acc {out['branches'][key]['mean']['acc']:.4f}")
        C.dump(res, args.out)
    for k, d in blocks:
        key = f"blocks@{k}+{d}"
        if key in out.get("branches", {}):
            continue
        log(f"-- {name} {key}")
        out.setdefault("branches", {})[key] = train_eval("blocks", k, d)
        log(f"   {key}: mean acc {out['branches'][key]['mean']['acc']:.4f}")
        C.dump(res, args.out)

    # --- zero-shot readouts (intent datasets only) ----------------------------------------------------
    if name not in ("banking77", "clinc150"):
        return
    labs = ds.tasks["intent"].labels
    inscope = [i for i, l in enumerate(labs) if l != "oos"]
    if args.smoke:
        present = sorted({e.y["intent"] for e in allx})
        inscope = [i for i in inscope if i in present]
    remap = {o: j for j, o in enumerate(inscope)}
    y_all = np.array([remap.get(e.y["intent"], -1) for e in allx])
    te = np.where(split_of == "test")[0]
    va = np.where(split_of == "val")[0]
    zs = out.setdefault("zeroshot", {})
    for tname, tmpl in TEMPLATES.items():
        ltexts = C.label_texts([labs[i] for i in inscope], tmpl)
        lcache = FeatureCache(trunk, ltexts, depths, 64)
        for pool in ("mean", "cls"):
            for d in depths:
                key = f"{tname}|{pool}|depth{d}"
                if key in zs:
                    continue
                L = C.pooled(lcache, d, list(range(len(ltexts))), pool)
                q_te, q_va = C.pooled(cache, d, list(te), pool), C.pooled(cache, d, list(va), pool)
                s_te, s_va = C.cosine_logits(q_te, L).numpy(), C.cosine_logits(q_va, L).numpy()
                zs[key] = zero_shot_eval(s_te, y_all[te], s_va, y_all[te] < 0, y_all[va] < 0)
        best = max((k for k in zs if k.startswith(tname)), key=lambda k: zs[k]["acc"])
        log(f"   zero-shot [{tname}] best {best}: acc {zs[best]['acc']:.4f}" +
            (f", oos auroc {zs[best]['oos']['auroc']:.3f}" if "oos" in zs[best] else ""))
        top = f"{tname}|{'cls' if 'gte' in trunk.base.lower() else 'mean'}|depth{n}"
        if top in zs:
            log(f"   zero-shot [{tname}] native pooling at the top ({top}): acc {zs[top]['acc']:.4f}")
    C.dump(res, args.out)

    if g is not None:
        texts_te = [allx[i].text for i in te]
        texts_va = [allx[i].text for i in va]
        names = [C.humanize(labs[i]) for i in inscope]
        gl = out.setdefault("gliclass_native", {})
        for mode in ("cached", "joint_chunk", "joint"):
            if mode in gl:
                continue
            t0 = time.time()
            if mode == "cached":
                s_te, s_va = gliclass_cached(trunk, g, texts_te, names), gliclass_cached(trunk, g, texts_va, names)
            else:
                ch = g["max_num_classes"] if mode == "joint_chunk" else None
                s_te = gliclass_joint(trunk, g, texts_te, names, ch)
                s_va = gliclass_joint(trunk, g, texts_va, names, ch)
            gl[mode] = zero_shot_eval(s_te, y_all[te], s_va, y_all[te] < 0, y_all[va] < 0)
            gl[mode]["seconds"] = round(time.time() - t0, 1)
            log(f"   gliclass {mode}: acc {gl[mode]['acc']:.4f}" +
                (f", oos auroc {gl[mode]['oos']['auroc']:.3f}" if "oos" in gl[mode] else "") + f" ({gl[mode]['seconds']}s)")
            C.dump(res, args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gte", help="modernbert | gte | gliclass | ettin | <any ModernBERT-architecture id>")
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150", "typed-cs"])
    ap.add_argument("--probe-depths", type=int, nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.datasets = ["banking77", "clinc150", "typed-cs"]
        args.out = args.out or f"results/tarski/explore_trunkswap_smoke_{args.base}.json"
    log = C.Log(args.out)
    g = None
    base = ALIASES.get(args.base, args.base)
    if args.base == "gliclass":
        g = gliclass_encoder()
        base = g["dir"]
    trunk = C.load_trunk(base, args.device)
    log(f"== trunk swap: {args.base} ({base}) on {trunk.device}, {trunk.n_layers} layers, hidden {trunk.hidden}")
    res = C.load_json(args.out) if not args.smoke else {}
    res.setdefault("config", vars(args) | {"base_path": base, "n_layers": trunk.n_layers})
    res.setdefault("datasets", {})
    for name in args.datasets:
        run_dataset(name, trunk, args, log, res, g)
        C.gpu_gc()
    C.dump(res, args.out)
    log("done")


if __name__ == "__main__":
    main()
