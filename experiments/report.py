"""Markdown tables from results/tarski/*.json, for docs/research/THESIS.md.

Usage: python -m experiments.report > results/tarski/tables.md
"""

import glob
import json
import os

R = "results/tarski"


def pct(x):
    return f"{100 * x:.1f}"


def sweep_table(path):
    res = json.load(open(path))
    name = os.path.basename(path)[len("sweep_"):-len(".json")]
    tasks = sorted({t for v in res.values() for t in v["tasks"]})
    full = res.get("full", {}).get("tasks", {})
    lines = [f"#### {name}", "",
             "| config | trunk layers run | branch params | " + " | ".join(tasks) + " | mean acc | mean ECE |",
             "|---|---:|---:|" + "---:|" * len(tasks) + "---:|---:|"]

    def key(item):
        n, v = item
        return (0 if v["kind"] == "probe" else (2 if n == "full" else 1), v["split"], v["depth"])

    for n, v in sorted(res.items(), key=key):
        cells = []
        for t in tasks:
            m = v["tasks"].get(t)
            if m is None:
                cells.append("")
                continue
            delta = f" ({100 * (m['acc'] - full[t]['acc']):+.1f})" if t in full and n != "full" else ""
            cells.append(pct(m["acc"]) + delta)
        params = max(m["params"] for m in v["tasks"].values())
        layers = v["split"] if n != "full" else 0
        lines.append(f"| {n} | {layers} | {params / 1e6:.2f}M | " + " | ".join(cells) +
                     f" | {pct(v['mean']['acc'])} | {v['mean']['ece']:.3f} |")
    if full:
        lines.append("")
        lines.append("Parenthesised: accuracy points relative to the full fine-tune of that task.")
    return "\n".join(lines)


def latency_table(path):
    res = json.load(open(path))
    out = [f"#### latency on {res['device']} (batch 1, median ms per message)", ""]
    for length, rows in res["per_message_ms"].items():
        ns = sorted(rows, key=int)
        cols = list(rows[ns[-1]])
        out += [f"{length} message ({res['tokens'][length]} tokens)", "",
                "| N decisions | " + " | ".join(cols) + " |", "|---:|" + "---:|" * len(cols)]
        for n in ns:
            out.append(f"| {n} | " + " | ".join(f"{rows[n].get(c, float('nan')):.1f}" if c in rows[n] else ""
                                                for c in cols) + " |")
        out.append("")
    out += ["| switch to a task | load ms | bytes |", "|---|---:|---:|"]
    for k, v in res["switch_ms"].items():
        out.append(f"| {k} | {v:.1f} | {res['bytes'][k] / 1e6:.2f} MB |")
    return "\n".join(out)


def main():
    parts = [sweep_table(p) for p in sorted(glob.glob(f"{R}/sweep_*.json"))]
    parts += [latency_table(p) for p in sorted(glob.glob(f"{R}/latency_*.json"))]
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
