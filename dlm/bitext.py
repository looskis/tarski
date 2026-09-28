"""Hard-label, multi-question test on the Bitext customer-support dataset (26,872 English support
messages, each with a category from 11 and an intent from 27, hard labels). Two-question canvas:
category and intent. Conditions: stock vs state-first format, given vs reversed order, random vs
mean slot; then the read policy (mean slot + best format/order + temperature, 2-fold) with paired
bootstrap intervals against the stock read.

  .venv/bin/python dlm/bitext.py --limit 250 --out results/dlm/bitext.jsonl
  .venv/bin/python dlm/bitext.py --report results/dlm/bitext.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm.backend import add_backend_arg, get_reader  # noqa: E402
from dlm.metrics import brier, ece  # noqa: E402
from dlm.state_first import prepare  # noqa: E402

CATEGORIES = {
    "ACCOUNT": "creating, editing, switching, recovering or deleting a customer account",
    "CANCEL": "cancelling an order or asking about cancellation fees",
    "CONTACT": "reaching customer service or a human agent",
    "DELIVERY": "delivery options and delivery periods",
    "FEEDBACK": "complaints and reviews",
    "INVOICE": "checking or obtaining an invoice",
    "ORDER": "placing, changing or tracking an order",
    "PAYMENT": "payment methods and payment issues",
    "REFUND": "refund policy, getting a refund, tracking a refund",
    "SHIPPING": "setting up or changing a shipping address",
    "SUBSCRIPTION": "newsletter subscription",
}
INTENTS = {
    "cancel_order": "cancel an order", "change_order": "change an existing order",
    "change_shipping_address": "change the shipping address", "check_cancellation_fee": "ask about cancellation fees",
    "check_invoice": "check an invoice", "check_payment_methods": "ask which payment methods are accepted",
    "check_refund_policy": "ask about the refund policy", "complaint": "file a complaint",
    "contact_customer_service": "contact customer service", "contact_human_agent": "speak to a human agent",
    "create_account": "create an account", "delete_account": "delete an account",
    "delivery_options": "ask about delivery options", "delivery_period": "ask how long delivery takes",
    "edit_account": "edit account details", "get_invoice": "obtain an invoice", "get_refund": "get a refund",
    "newsletter_subscription": "subscribe to or unsubscribe from the newsletter", "payment_issue": "report a payment problem",
    "place_order": "place an order", "recover_password": "recover a password", "registration_problems": "report a problem registering",
    "review": "leave a review", "set_up_shipping_address": "set up a shipping address", "switch_account": "switch to another account",
    "track_order": "track an order", "track_refund": "track a refund",
}
QUESTIONS = {
    "category": {"type": "choice", "instructions": "Which category does this customer message belong to?", "criteria": CATEGORIES},
    "intent": {"type": "choice", "instructions": "What does the customer want to do?", "criteria": INTENTS},
}
FORMATS = ("stock", "state_first")
ORDERS = {"given": ["category", "intent"], "reversed": ["intent", "category"]}


def load(limit, seed=0):
    from datasets import load_dataset
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    idx = random.Random(seed).sample(range(len(ds)), limit)
    return [{"id": f"bitext/{i}", "state": ds[i]["instruction"], "gold": {"category": ds[i]["category"], "intent": ds[i]["intent"]}} for i in idx]


def run(a):
    recs = load(a.limit)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} messages, {len(todo)} to run; ETA ~{len(todo) * 9 / 60:.0f} min on the laptop", flush=True)
    r = get_reader(a.backend)
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            out = {q: {} for q in QUESTIONS}
            for oname, order in ORDERS.items():
                qdict = {q: QUESTIONS[q] for q in order}
                for fmt in FORMATS:
                    prep = prepare(r, rec["state"], qdict, fmt)
                    keys = [q["key"] for q in prep["qs"]]
                    for mode in ("random", "mean"):
                        for key, p in zip(keys, r.read(prep, mode, seed=0)):
                            out[key][f"{fmt}|{oname}|{mode}"] = p
                    if n == 1 and oname == "given":
                        print(f"  {fmt}: prompt {len(prep['prompt'])} tokens, canvas {prep['width']}", flush=True)
            base = r.prepare(rec["state"], QUESTIONS)
            row = {"id": rec["id"], "state": rec["state"], "questions": []}
            for q in base["qs"]:
                row["questions"].append({"qid": q["key"], "type": q["type"], "labels": [c[0] for c in q["choices"]],
                                         "gold": rec["gold"][q["key"]], "reads": out[q["key"]]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def report(path):
    from dlm.calibrate import fit, scale
    rows = [json.loads(l) for l in open(path) if l.strip()]
    conds = [f"{f}|{o}|{m}" for f in FORMATS for o in ORDERS for m in ("random", "mean")]
    print(f"{len(rows)} messages, {2 * len(rows)} slots (hard labels)")
    print(f"{'condition':26s} {'category':>22s} {'intent':>22s} {'both':>22s}")
    per = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for q in row["questions"]:
            y = q["labels"].index(q["gold"])
            for c in conds:
                p = q["reads"][c]; top = max(range(len(p)), key=p.__getitem__)
                per[c][q["qid"]].append((top == y, p[top], brier(p, y)))
    def stats(v): return (sum(x[0] for x in v) / len(v), sum(x[2] for x in v) / len(v), ece([(x[1], x[0]) for x in v]))
    for c in conds:
        cells = []
        for k in ("category", "intent", "both"):
            v = per[c]["category"] + per[c]["intent"] if k == "both" else per[c][k]
            a, b, e = stats(v); cells.append(f"{a:.3f}/{b:.3f}/{e:.3f}")
        print(f"{c:26s} " + " ".join(f"{x:>22s}" for x in cells))
    print("  (cells: acc / Brier / ECE)")
    # read policy: mean slot, state-first, given order, + temperature 2-fold; vs stock default
    best = "state_first|given|mean"
    items = defaultdict(list)
    for si, row in enumerate(rows):
        for q in row["questions"]:
            items[q["qid"]].append((q["reads"][best], tuple(q["labels"]), q["labels"].index(q["gold"]), si))
    outs = []
    for qid, its in items.items():
        for h in (0, 1):
            fit_items = [x[:3] for x in its if x[3] % 2 != h]; sc = [x for x in its if x[3] % 2 == h]
            priors, T1, _ = fit(fit_items)
            for p, labels, y, si in sc:
                q = scale(p, T1, 0.0, priors[labels]); top = max(range(len(q)), key=q.__getitem__)
                outs.append((top == y, q[top], brier(q, y)))
    a, b, e = stats(outs)
    print(f"\nread policy (state-first, mean slot, temperature 2-fold): acc {a:.3f} brier {b:.3f} ece {e:.3f}")
    # paired bootstrap: policy ranking vs stock default, by state
    random.seed(0)
    ps = []
    for row in rows:
        d = 0
        for q in row["questions"]:
            y = q["labels"].index(q["gold"])
            pa, pb = q["reads"][best], q["reads"]["stock|given|random"]
            d += (max(range(len(pa)), key=pa.__getitem__) == y) - (max(range(len(pb)), key=pb.__getitem__) == y)
        ps.append(d)
    n = 2 * len(rows); base = sum(ps) / n; bs = sorted(sum(random.choice(ps) for _ in ps) / n for _ in range(4000))
    print(f"policy - stock default, accuracy: {base:+.3f}  95% CI [{bs[100]:+.3f}, {bs[3899]:+.3f}]  ({len(rows)} states, {n} slots)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--out", default="results/dlm/bitext.jsonl")
    ap.add_argument("--report", default=None)
    add_backend_arg(ap)
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(a)


if __name__ == "__main__":
    main()
