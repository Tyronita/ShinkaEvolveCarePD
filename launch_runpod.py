"""
launch_runpod.py — Provision a RunPod network volume and GPU pod for CARE-PD ShinkaEvolve.

Usage:
    python launch_runpod.py                  # trial run (30 gens)
    python launch_runpod.py --generations 100 --cost-cap 50
    python launch_runpod.py --status         # check existing pod
    python launch_runpod.py --stop <pod_id>  # stop pod
"""

import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["RUNPOD_API_KEY"]
GITHUB_REPO = "Tyronita/ShinkaEvolveCarePD"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
REST_BASE = "https://rest.runpod.io/v1"

# ── GPU priority lists ────────────────────────────────────────────────────────
GPU_PRIORITY_BIG = [        # --big flag: A100/H100/L40S first, wide fallback
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA L40",
    "NVIDIA A40",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
]
GPU_PRIORITY = [            # default: RTX 4090 / A5000
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4000",
]

DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

BOOTSTRAP_CMD = (
    "apt-get install -y git -qq 2>/dev/null; "
    "git clone https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git "
    "    /workspace/ShinkaEvolveCarePD 2>/dev/null "
    "|| (cd /workspace/ShinkaEvolveCarePD && git pull --ff-only 2>/dev/null || true); "
    "bash /workspace/ShinkaEvolveCarePD/runpod_startup.sh 2>&1 | tee /workspace/startup.log"
)


def _get(path, **kwargs):
    r = requests.get(f"{REST_BASE}/{path}", headers=HEADERS, **kwargs)
    r.raise_for_status()
    return r.json()


def _post(path, body):
    r = requests.post(f"{REST_BASE}/{path}", headers=HEADERS, json=body)
    if not r.ok:
        print(f"[error] {r.status_code}: {r.text}")
    r.raise_for_status()
    return r.json()


def _delete(path):
    r = requests.delete(f"{REST_BASE}/{path}", headers=HEADERS)
    r.raise_for_status()
    return r.json()


# ── Network volume ────────────────────────────────────────────────────────────

def get_or_create_volume(name="care-pd-workspace", size_gb=50) -> str:
    """Return existing network volume ID, or create a new one."""
    try:
        vols = _get("networkvolumes")
        for v in vols:
            if v.get("name") == name:
                print(f"[volume] Using existing volume: {v['id']} ({v['name']}, {v['size']}GB)")
                return v["id"]
    except Exception as e:
        print(f"[volume] Could not list volumes: {e}")

    print(f"[volume] Creating new network volume '{name}' ({size_gb}GB)...")
    # Use a common US datacenter — TX-3 or KS-2
    for dc in ["US-TX-3", "US-KS-2", "US-CA-3", "EU-RO-1"]:
        try:
            vol = _post("networkvolumes", {"name": name, "size": size_gb, "dataCenterId": dc})
            print(f"[volume] Created: {vol['id']} in {dc}")
            return vol["id"]
        except Exception as e:
            print(f"[volume] {dc} failed: {e}, trying next...")

    raise RuntimeError("Could not create network volume in any datacenter")


# ── Pod ───────────────────────────────────────────────────────────────────────

def launch_pod(
    volume_id: str,
    num_generations: int = 30,
    max_eval_jobs: int = 2,
    max_api_costs: float = 12.0,
    task_name: str = "care_pd_task",
    big_gpu: bool = False,
) -> dict:
    env = {
        # Secrets — must be created at console.runpod.io/user/secrets first
        "ANTHROPIC_API_KEY":  "{{ RUNPOD_SECRET_ANTHROPIC_API_KEY }}",
        "GEMINI_API_KEY":     "{{ RUNPOD_SECRET_GEMINI_API_KEY }}",
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
        "HF_TOKEN":           "{{ RUNPOD_SECRET_HF_TOKEN }}",
        "GITHUB_TOKEN":       "{{ RUNPOD_SECRET_GITHUB_TOKEN }}",
        "GITHUB_EMAIL":       "{{ RUNPOD_SECRET_GITHUB_EMAIL }}",
        # Plain config (not sensitive)
        "GITHUB_REPO":        GITHUB_REPO,
        "TASK_NAME":          task_name,
        "NUM_GENERATIONS":    str(num_generations),
        "MAX_EVAL_JOBS":      str(max_eval_jobs),
        "MAX_PROPOSAL_JOBS":  str(max_eval_jobs),
        "MAX_API_COSTS":      str(max_api_costs),
        "WEBUI_PORT":         "8080",
        "DOCS_PORT":          "8888",
    }

    gpu_list = GPU_PRIORITY_BIG if big_gpu else GPU_PRIORITY
    pod_config = {
        "name":               f"care-pd-{task_name}",
        "imageName":          DOCKER_IMAGE,
        "cloudType":          "SECURE",
        "computeType":        "GPU",
        "gpuCount":           1,
        "gpuTypeIds":         gpu_list,
        "gpuTypePriority":    "custom",  # try in order
        "containerDiskInGb":  80,        # no network volume — store dataset + results on disk
        "ports":              ["22/tcp", "8080/http", "8888/http"],
        "env":                env,
        "dockerStartCmd":     ["bash", "-c", BOOTSTRAP_CMD],
    }

    print(f"[pod] Launching pod (gens={num_generations}, max_cost=${max_api_costs})...")
    pod = _post("pods", pod_config)
    return pod


def get_pod_status(pod_id: str) -> dict:
    return _get(f"pods/{pod_id}")


def stop_pod(pod_id: str):
    return _post(f"pods/{pod_id}/stop", {})


def print_pod_info(pod: dict):
    pod_id = pod.get("id", "?")
    status = pod.get("desiredStatus") or pod.get("status", "?")
    machine = pod.get("machine") or {}
    ip = machine.get("podHostId") or pod.get("runtime", {}).get("ports", [{}])[0].get("ip", "pending")

    print()
    print("=" * 60)
    print(f"  Pod ID:   {pod_id}")
    print(f"  Status:   {status}")
    print(f"  GPU:      {pod.get('gpuDisplayName', '?')}")
    print(f"  IP:       {ip}")
    print()

    # SSH info
    ports = pod.get("runtime", {}).get("ports", [])
    ssh_port = next((p.get("publicPort") for p in ports if p.get("privatePort") == 22), None)
    if ssh_port:
        print(f"  SSH:      ssh root@{ip} -p {ssh_port} -i ~/.ssh/your_key")
        print(f"  Web UI:   (tunnel) ssh -L 8080:localhost:8080 root@{ip} -p {ssh_port}")
        print(f"  Docs:     (tunnel) ssh -L 8888:localhost:8888 root@{ip} -p {ssh_port}")
    else:
        print("  SSH:      (waiting for IP — check RunPod console)")
    print()
    print(f"  Logs:     tail -f /workspace/startup.log  (once SSH'd in)")
    print(f"  Stop:     python launch_runpod.py --stop {pod_id}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Manage CARE-PD RunPod pod")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--cost-cap", type=float, default=12.0)
    parser.add_argument("--eval-jobs", type=int, default=2)
    parser.add_argument("--task", default="care_pd_task", help="TASK_NAME to run (care_pd_task | care_pd_task_v2)")
    parser.add_argument("--big", action="store_true", help="Use bigger GPU (A100/A6000 priority)")
    parser.add_argument("--status", metavar="POD_ID", help="Check pod status")
    parser.add_argument("--stop", metavar="POD_ID", help="Stop a pod")
    parser.add_argument("--list", action="store_true", help="List all pods")
    args = parser.parse_args()

    if args.status:
        pod = get_pod_status(args.status)
        print_pod_info(pod)
        return

    if args.stop:
        print(f"Stopping pod {args.stop}...")
        stop_pod(args.stop)
        print("Stop requested.")
        return

    if args.list:
        pods = _get("pods")
        if not pods:
            print("No pods found.")
        for p in pods:
            print(f"  {p['id']}  {p.get('name','?'):30s}  {p.get('desiredStatus','?')}")
        return

    # Launch (no network volume — use large container disk, git handles persistence)
    print(f"[launch] task={args.task}  gens={args.generations}  cost_cap=${args.cost_cap}  big_gpu={args.big}")
    pod = launch_pod(
        volume_id=None,
        num_generations=args.generations,
        max_eval_jobs=args.eval_jobs,
        max_api_costs=args.cost_cap,
        task_name=args.task,
        big_gpu=args.big,
    )
    print_pod_info(pod)

    print("NOTE: RunPod secrets must be set at console.runpod.io/user/secrets")
    print("      Required: ANTHROPIC_API_KEY, GEMINI_API_KEY, HF_TOKEN,")
    print("                GITHUB_TOKEN, GITHUB_EMAIL")


if __name__ == "__main__":
    main()
