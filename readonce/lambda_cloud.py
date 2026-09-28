"""Minimal Lambda Cloud helper for short experiment runs.

  python -m readonce.lambda_cloud launch     # wait for capacity, launch the first acceptable 1-GPU type
  python -m readonce.lambda_cloud status
  python -m readonce.lambda_cloud terminate  # terminates the instance recorded in the state file
  python -m readonce.lambda_cloud watchdog --hours 4   # terminate after N hours (run under nohup)

Reads LAMBDALABS_API_KEY from the environment (e.g. `set -a; source .env.local; set +a`).
The launched instance id is recorded in $LAMBDA_STATE (default ./.lambda_instance.json) so it is
never lost; instances bill until terminated, not until shut down.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

API = "https://cloud.lambda.ai/api/v1"
# fastest-per-dollar first; GH200 is aarch64 and B200/multi-GPU are overkill for this workload
PREFERENCE = ["gpu_1x_h100_pcie", "gpu_1x_a100_sxm4", "gpu_1x_a100", "gpu_1x_h100_sxm5",
              "gpu_1x_a6000", "gpu_1x_a10", "gpu_1x_rtx6000"]
STATE = os.environ.get("LAMBDA_STATE", ".lambda_instance.json")


def call(method, path, body=None):
    key = os.environ["LAMBDALABS_API_KEY"]
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{key}:".encode()).decode())
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "tarski-readonce/0.1")   # the default Python-urllib agent is rejected
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return json.loads(body or b"{}")
        except json.JSONDecodeError:
            return {"error": {"code": e.code, "message": body[:200].decode(errors="replace")}}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": {"message": str(e)}}


def available():
    r = call("GET", "/instance-types")
    if "error" in r:
        print("instance-types error:", r["error"], flush=True)
    data = r.get("data", {})
    out = {}
    for name, v in data.items():
        regions = [r["name"] for r in v.get("regions_with_capacity_available", [])]
        if regions:
            out[name] = (regions, v["instance_type"]["price_cents_per_hour"] / 100)
    return out


def launch(args):
    if os.path.exists(STATE):
        sys.exit(f"{STATE} already records an instance; terminate it first")
    deadline = time.time() + args.wait_minutes * 60
    while True:
        avail = available()
        choice = next((t for t in PREFERENCE if t in avail and avail[t][1] <= args.max_price), None)
        if choice:
            region = avail[choice][0][0]
            r = call("POST", "/instance-operations/launch", {
                "region_name": region, "instance_type_name": choice, "ssh_key_names": [args.ssh_key],
                "quantity": 1, "name": args.name})
            ids = r.get("data", {}).get("instance_ids")
            if ids:
                json.dump({"id": ids[0], "type": choice, "region": region, "price": avail[choice][1],
                           "launched_at": time.time()}, open(STATE, "w"))
                print(f"launched {choice} in {region} (${avail[choice][1]:.2f}/h): {ids[0]}", flush=True)
                break
            print(f"launch of {choice} failed: {r.get('error')}", flush=True)
        if time.time() > deadline:
            sys.exit("no capacity within the wait window")
        print(time.strftime("%H:%M:%S"), "no acceptable capacity; available:",
              {k: v[1] for k, v in avail.items()} or "none", flush=True)
        time.sleep(args.poll_seconds)
    wait_active()


def wait_active():
    st = json.load(open(STATE))
    while True:
        inst = call("GET", f"/instances/{st['id']}").get("data", {})
        if inst.get("status") == "active" and inst.get("ip"):
            st["ip"] = inst["ip"]
            json.dump(st, open(STATE, "w"))
            print(f"active: ubuntu@{inst['ip']}", flush=True)
            return
        if inst.get("status") in ("terminated", "unhealthy"):
            sys.exit(f"instance is {inst.get('status')}")
        time.sleep(10)


def status(_):
    if not os.path.exists(STATE):
        print("no recorded instance")
    else:
        st = json.load(open(STATE))
        inst = call("GET", f"/instances/{st['id']}").get("data", {})
        hours = (time.time() - st["launched_at"]) / 3600
        print(f"{st['type']} {inst.get('status')} ip={inst.get('ip')} up {hours:.2f} h "
              f"(~${hours * st['price']:.2f})")
    for i in call("GET", "/instances").get("data", []):
        print("account instance:", i["id"], i.get("name"), i["instance_type"]["name"], i["status"])


def terminate(_):
    if not os.path.exists(STATE):
        sys.exit("no recorded instance")
    st = json.load(open(STATE))
    r = call("POST", "/instance-operations/terminate", {"instance_ids": [st["id"]]})
    done = [i["id"] for i in r.get("data", {}).get("terminated_instances", [])]
    if st["id"] in done:
        hours = (time.time() - st["launched_at"]) / 3600
        print(f"terminated {st['id']} after {hours:.2f} h (~${hours * st['price']:.2f})")
        os.remove(STATE)
    else:
        sys.exit(f"terminate failed: {r}")


def watchdog(args):
    """Terminate the recorded instance after `hours`, a backstop in case the session ends early."""
    st = json.load(open(STATE))
    while time.time() < st["launched_at"] + args.hours * 3600:
        time.sleep(60)
        if not os.path.exists(STATE):
            return
    terminate(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--ssh-key", default="claude-tarski-readonce")
    p.add_argument("--name", default="tarski-readonce")
    p.add_argument("--max-price", type=float, default=4.5)
    p.add_argument("--wait-minutes", type=float, default=180)
    p.add_argument("--poll-seconds", type=float, default=60)
    sub.add_parser("status")
    sub.add_parser("terminate")
    w = sub.add_parser("watchdog")
    w.add_argument("--hours", type=float, default=4)
    a = ap.parse_args()
    {"launch": launch, "status": status, "terminate": terminate, "watchdog": watchdog}[a.cmd](a)
