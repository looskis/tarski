"""Submit the jobs listed in explore/QUEUE_*.txt to the Lambda job runners.

Each line: `name | A10 or A100 | est. minutes | command`. Submitted names are recorded in
explore/.submitted so re-running is safe. Syncs code to both instances first.

Usage: LAMBDA_SSH_KEY=... python experiments/submit_queue.py explore/QUEUE_skeptic.txt [...]
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATES = {"A10": ".lambda_instance.json", "A100": ".lambda_instance_fast.json"}
DONE = os.path.join(ROOT, "explore", ".submitted")


def helper(state, *args):
    env = {**os.environ, "LAMBDA_STATE": state}
    return subprocess.run(["bash", "experiments/lambda.sh", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def main(paths):
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    for state in STATES.values():
        r = helper(state, "sync")
        if r.returncode:
            sys.exit(f"sync to {state} failed: {r.stderr[-300:]}")
    for path in paths:
        for line in open(path):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|", 3)]
            if len(parts) != 4:
                print(f"skip malformed line in {path}: {line.strip()[:80]}")
                continue
            name, gpu, _minutes, cmd = parts
            if name.lower() == "name" or name in done:          # header line, or already submitted
                continue
            state = STATES.get(gpu.upper(), STATES["A10"])
            r = helper(state, "submit", name, cmd)
            print(f"{name} -> {gpu}: {(r.stdout or r.stderr).strip()[-80:]}")
            if r.returncode == 0:
                done.add(name)
                with open(DONE, "a") as f:
                    f.write(name + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
