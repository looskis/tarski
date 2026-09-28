"""Poll Lambda for a faster instance than the A10 and launch the first one available.

Reuses readonce/lambda_cloud.py (API calls, launch, wait-for-active) with its own preference list and
its own state file, so the A10 recorded in .lambda_instance.json is untouched.

Usage: set -a; . ./.env.local; set +a; python experiments/lambda_poll_fast.py --wait-minutes 150
Then: LAMBDA_STATE=.lambda_instance_fast.json python -m readonce.lambda_cloud status | terminate
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import readonce.lambda_cloud as lc  # noqa: E402

lc.STATE = os.environ.get("LAMBDA_STATE", ".lambda_instance_fast.json")
# fastest for this workload first; 2x A100 runs the experiment lanes on separate GPUs.
# GH200 (aarch64 CPU) and B200 are left out on purpose.
lc.PREFERENCE = ["gpu_1x_h100_sxm5", "gpu_2x_a100", "gpu_1x_h100_pcie", "gpu_1x_a100_sxm4", "gpu_1x_a100"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh-key", default="claude-tarski-readonce")
    ap.add_argument("--name", default="tarski-fast")
    ap.add_argument("--max-price", type=float, default=4.29)
    ap.add_argument("--wait-minutes", type=float, default=150)
    ap.add_argument("--poll-seconds", type=float, default=60)
    lc.launch(ap.parse_args())
