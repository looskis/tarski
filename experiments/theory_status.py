"""Which queued experiments have results? Lists every entry in explore/QUEUE_*.txt with its status.

A result counts as present when results/tarski/{lambda,lambda_a100,.}/explore_<name>.json (or any JSON
named in the command's --out) exists. Run `experiments/lambda.sh pull` (both instances) first.

Usage: python experiments/theory_status.py [--missing]
"""

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ["results/tarski/lambda", "results/tarski/lambda_a100", "results/tarski"]


def main(missing_only=False):
    rows = []
    for q in sorted(glob.glob(os.path.join(ROOT, "explore", "QUEUE_*.txt"))):
        lens = os.path.basename(q)[len("QUEUE_"):-len(".txt")]
        for line in open(q):
            parts = [p.strip() for p in line.split("|", 3)]
            if len(parts) != 4 or line.lstrip().startswith("#") or parts[0].lower() == "name":
                continue
            name, gpu, minutes, cmd = parts
            m = re.search(r"--out\s+(\S+)", cmd)
            out = os.path.basename(m.group(1)) if m else f"explore_{name}.json"
            found = [d for d in DIRS if os.path.exists(os.path.join(ROOT, d, out))]
            run_log = [d for d in DIRS if os.path.exists(os.path.join(ROOT, d, f"run_{name}.out"))]
            status = "done" if found else ("ran, no result (check run log)" if run_log else "pending")
            rows.append((lens, name, gpu, status, found[0] if found else ""))
    done = sum(r[3] == "done" for r in rows)
    print(f"{done}/{len(rows)} queued experiments have results")
    for lens, name, gpu, status, where in rows:
        if missing_only and status == "done":
            continue
        print(f"  {lens:10s} {name:28s} {gpu:5s} {status:30s} {where}")


if __name__ == "__main__":
    main("--missing" in sys.argv)
