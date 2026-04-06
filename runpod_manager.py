"""
RunPod Manager for ShinkaEvolve CARE-PD
========================================
Reads from .env. Subcommands:

  python runpod_manager.py status          # pod status + balance
  python runpod_manager.py logs            # stream pod logs (polls every 10s)
  python runpod_manager.py deploy          # spin up a new pod
  python runpod_manager.py stop            # stop the pod
  python runpod_manager.py terminate       # terminate the pod (deletes it)
  python runpod_manager.py ssh <cmd>       # run a shell command on the pod
  python runpod_manager.py watch           # loop: status + tail logs every 30s
  python runpod_manager.py leaderboard     # pull + print latest leaderboard.csv from pod
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

RUNPOD_API_KEY  = os.environ.get("RUNPOD_API_KEY", "")
GITHUB_TOKEN    = os.environ.get("GITHUB_TOKEN", "")
GITHUB_EMAIL    = os.environ.get("GITHUB_EMAIL", "")
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY      = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
HF_TOKEN        = os.environ.get("HF_TOKEN", "")

GITHUB_REPO     = "Tyronita/ShinkaEvolveCarePD"
API_URL         = "https://api.runpod.io/graphql"
POD_NAME        = "care-pd-shinka"

# ── GraphQL helper ────────────────────────────────────────────────────────────

def gql(query: str, variables: dict = None) -> dict:
    import subprocess, json as _json
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    body = _json.dumps(payload)
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", API_URL,
         "-H", "Content-Type: application/json",
         "-H", f"Authorization: Bearer {RUNPOD_API_KEY}",
         "-d", body],
        capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"curl error: {result.stderr}")
    data = _json.loads(result.stdout)
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data.get("data", {})

# ── Pod queries ───────────────────────────────────────────────────────────────

def get_pods() -> list:
    data = gql("""{ myself { pods {
        id name desiredStatus
        runtime { uptimeInSeconds
                  ports { ip isIpPublic privatePort publicPort type } }
        machine { podHostId }
        gpuCount imageName
    } clientBalance } }""")
    return data.get("myself", {}).get("pods", []), \
           data.get("myself", {}).get("clientBalance", 0)


def get_pod() -> dict | None:
    pods, _ = get_pods()
    for p in pods:
        if POD_NAME in p.get("name", ""):
            return p
    return pods[0] if pods else None


def get_ssh_info(pod: dict) -> tuple[str, int] | None:
    """Return (ip, port) for SSH if available."""
    runtime = pod.get("runtime") or {}
    for port in (runtime.get("ports") or []):
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return port["ip"], port["publicPort"]
    return None


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_status():
    pods, balance = get_pods()
    print(f"\nBalance: ${balance:.4f}")
    if not pods:
        print("No pods running.")
        return
    for pod in pods:
        rt      = pod.get("runtime") or {}
        uptime  = rt.get("uptimeInSeconds", 0)
        hours   = uptime // 3600
        mins    = (uptime % 3600) // 60
        ssh     = get_ssh_info(pod)
        ssh_str = f"  SSH: ssh root@{ssh[0]} -p {ssh[1]}" if ssh else "  SSH: not yet available"
        ports   = [p for p in (rt.get("ports") or []) if p.get("isIpPublic")]
        http_ports = [f"port {p['publicPort']}→{p['privatePort']}" for p in ports if p.get("type") == "http"]
        print(f"\nPod: {pod['name']}  ({pod['id']})")
        print(f"  Status:  {pod['desiredStatus']}")
        print(f"  Image:   {pod.get('imageName', '?')}")
        print(f"  Uptime:  {hours}h {mins}m")
        print(ssh_str)
        if http_ports:
            print(f"  HTTP:   {', '.join(http_ports)}")
    print()


def cmd_logs(tail_lines: int = 100, follow: bool = False, interval: int = 15):
    """Get pod logs via SSH or RunPod log endpoint."""
    pod = get_pod()
    if not pod:
        print("No pod found.")
        return

    ssh = get_ssh_info(pod)
    if not ssh:
        print("Pod not ready for SSH yet — waiting...")
        for _ in range(20):
            time.sleep(15)
            pod = get_pod()
            ssh = get_ssh_info(pod) if pod else None
            if ssh:
                break
        if not ssh:
            print("SSH still not available after 5 minutes.")
            return

    ip, port = ssh
    print(f"\nConnecting to {ip}:{port} ...\n{'='*60}")

    def run_ssh(command: str) -> str:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(port), f"root@{ip}", command],
            capture_output=True, text=True, timeout=30)
        return result.stdout + result.stderr

    if not follow:
        # One-shot tail
        out = run_ssh(f"tail -n {tail_lines} /proc/1/fd/1 2>/dev/null || "
                      f"journalctl -n {tail_lines} --no-pager 2>/dev/null || "
                      f"cat /root/startup.log 2>/dev/null || echo 'No logs found'")
        print(out)
    else:
        # Poll loop
        print(f"Polling logs every {interval}s (Ctrl+C to stop)...\n")
        seen = set()
        while True:
            try:
                # Try to get the startup log + shinka output
                out = run_ssh(
                    "tail -n 200 /root/startup.log 2>/dev/null; "
                    "echo '---SHINKA---'; "
                    "ls /workspace/ShinkaEvolveCarePD/care_pd_task/leaderboard.csv 2>/dev/null "
                    "  && tail -n 20 /workspace/ShinkaEvolveCarePD/care_pd_task/leaderboard.csv "
                    "  || echo 'No leaderboard yet'"
                )
                lines = out.splitlines()
                new_lines = [l for l in lines if l not in seen]
                if new_lines:
                    for l in new_lines:
                        print(l)
                    seen.update(lines)
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"[warn] {e}")
                time.sleep(interval)


def cmd_ssh(command: str):
    pod = get_pod()
    if not pod:
        print("No pod found.")
        return
    ssh = get_ssh_info(pod)
    if not ssh:
        print("SSH not available yet.")
        return
    ip, port = ssh
    print(f"[{ip}:{port}] $ {command}\n")
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-p", str(port), f"root@{ip}", command],
        text=True)
    return result.returncode


def cmd_watch(interval: int = 30):
    """Loop: print status + latest leaderboard rows every N seconds."""
    print(f"Watching pod every {interval}s (Ctrl+C to stop)...\n")
    while True:
        try:
            ts = time.strftime("%H:%M:%S")
            print(f"\n{'='*60}  {ts}")
            pod = get_pod()
            if not pod:
                print("No pod running.")
            else:
                rt     = pod.get("runtime") or {}
                uptime = rt.get("uptimeInSeconds", 0)
                ssh    = get_ssh_info(pod)
                print(f"Pod: {pod['name']}  status={pod['desiredStatus']}  "
                      f"uptime={uptime//3600}h{(uptime%3600)//60}m")
                if ssh:
                    ip, port = ssh
                    # Get latest leaderboard + last shinka log line
                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
                         "-p", str(port), f"root@{ip}",
                         "echo '--- LEADERBOARD (last 10) ---'; "
                         "tail -n 10 /workspace/ShinkaEvolveCarePD/care_pd_task/leaderboard.csv 2>/dev/null || echo 'none yet'; "
                         "echo '--- LAST LOG LINES ---'; "
                         "tail -n 15 /root/startup.log 2>/dev/null || echo 'no log'"],
                        capture_output=True, text=True, timeout=15)
                    print(result.stdout)
                    if result.stderr:
                        print(result.stderr[:200])
                else:
                    print("SSH not ready yet.")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"[error] {e}")
            time.sleep(interval)


def cmd_leaderboard():
    """Pull leaderboard.csv from pod and display it."""
    pod = get_pod()
    if not pod:
        print("No pod running.")
        return
    ssh = get_ssh_info(pod)
    if not ssh:
        print("SSH not available.")
        return
    ip, port = ssh
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-p", str(port), f"root@{ip}",
         "cat /workspace/ShinkaEvolveCarePD/care_pd_task/leaderboard.csv 2>/dev/null || echo 'not found'"],
        capture_output=True, text=True, timeout=20)
    import csv, io
    content = result.stdout.strip()
    if "not found" in content or not content:
        print("No leaderboard yet.")
        return
    rows = list(csv.DictReader(io.StringIO(content)))
    rows.sort(key=lambda r: float(r.get("macro_f1", 0) or 0), reverse=True)
    print(f"\n{'='*72}")
    print(f"  LEADERBOARD — {len(rows)} genomes  (from pod)")
    print(f"{'='*72}")
    print(f"{'Rank':<5} {'Genome':<10} {'macro_F1':>9} {'F1-sev':>8} {'F1-mild':>8} {'time(s)':>8}")
    print("-"*52)
    for i, r in enumerate(rows[:20], 1):
        print(f"{i:<5} {r.get('genome_id','?'):<10} "
              f"{float(r.get('macro_f1',0) or 0):>9.4f} "
              f"{str(r.get('f1_class2','')):>8} "
              f"{str(r.get('f1_class1','')):>8} "
              f"{str(r.get('eval_time_s','')):>8}")
    print()


def cmd_deploy(gpu_type: str = "NVIDIA RTX A5000", gpu_count: int = 1):
    """Spin up a new pod."""
    env_vars = [
        {"key": "ANTHROPIC_API_KEY",  "value": ANTHROPIC_KEY},
        {"key": "GEMINI_API_KEY",     "value": GEMINI_KEY},
        {"key": "OPENROUTER_API_KEY", "value": OPENROUTER_KEY},
        {"key": "HF_TOKEN",           "value": HF_TOKEN},
        {"key": "GITHUB_TOKEN",       "value": GITHUB_TOKEN},
        {"key": "GITHUB_EMAIL",       "value": GITHUB_EMAIL},
        {"key": "GITHUB_REPO",        "value": GITHUB_REPO},
        {"key": "NUM_GENERATIONS",    "value": "100"},
        {"key": "MAX_EVAL_JOBS",      "value": "4"},
        {"key": "MAX_API_COSTS",      "value": "40.0"},
        {"key": "SMOKE_ONLY",         "value": "false"},
    ]

    IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

    # Build a dockerArgs string with NO shell operators (they get tokenized).
    # Strategy: base64-encode a Python bootstrap script, then run it via:
    #   python3 -c exec(__import__('base64').b64decode('BASE64').decode())
    # Tokenizes to exactly 3 args: ["python3", "-c", "exec(...)"] — no spaces
    # in the exec(...) string, so no further splitting occurs.
    import base64 as _b64
    _bootstrap = (
        "import os,subprocess\n"
        "t=os.environ.get('GITHUB_TOKEN','')\n"
        "r='Tyronita/ShinkaEvolveCarePD'\n"
        "d='/workspace/ShinkaEvolveCarePD'\n"
        "u=f'https://{t}@github.com/{r}.git' if t else f'https://github.com/{r}.git'\n"
        "subprocess.run(['git','clone',u,d] if not os.path.exists(d+'/.git') else ['git','-C',d,'pull','--ff-only'],check=False)\n"
        "os.execv('/bin/bash',['/bin/bash',d+'/runpod_startup.sh'])\n"
    )
    _enc = _b64.b64encode(_bootstrap.encode()).decode()
    # dockerArgs is split on ALL spaces — the -c code must have ZERO spaces.
    # Use __import__(chr()...) to avoid both import statements (need space) and
    # string literals (quotes get stripped by shell). sys.argv[1] carries payload.
    # Tokens: ["python3", "-c", "exec(...no spaces...)", "BASE64"]
    _sys  = "+".join(f"chr({ord(c)})" for c in "sys")
    _b64m = "+".join(f"chr({ord(c)})" for c in "base64")
    _code = f"exec(__import__({_b64m}).b64decode(__import__({_sys}).argv[1]))"
    STARTUP_CMD = f"python3 -c {_code} {_enc}"

    mutation = """
    mutation DeployPod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id name desiredStatus imageName
        machine { podHostId }
      }
    }"""

    variables = {"input": {
        "cloudType": "SECURE",
        "gpuCount": gpu_count,
        "gpuTypeId": gpu_type,
        "name": POD_NAME,
        "imageName": IMAGE,
        "dockerArgs": STARTUP_CMD,
        "ports": "22/tcp,8080/http,8888/http",
        "volumeInGb": 50,
        "containerDiskInGb": 50,
        "volumeMountPath": "/workspace",
        "minVcpuCount": 4,
        "minMemoryInGb": 16,
        "startSsh": True,
        "env": env_vars,
    }}

    print(f"Deploying pod '{POD_NAME}' on {gpu_type} x{gpu_count}...")
    print(f"  Image:   {IMAGE}")
    print(f"  Startup: {STARTUP_CMD}")
    print(f"  Env vars: {len(env_vars)} set")
    print(f"  Logs visible in RunPod dashboard -> pod logs\n")

    result = gql(mutation, variables)
    pod = result.get("podFindAndDeployOnDemand", {})
    print(f"Deployed! Pod ID: {pod.get('id')}")
    print(f"Status: {pod.get('desiredStatus')}")
    print(f"\nRun: python runpod_manager.py logs   (to stream logs)")
    cmd_status()


def cmd_bootstrap():
    """
    Wait for SSH then run the full startup sequence on the pod.
    Clones the repo and runs runpod_startup.sh inside a nohup session
    so it keeps running after this script exits.
    """
    print("Waiting for pod SSH...")
    pod = None
    ssh = None
    for _ in range(40):
        pod = get_pod()
        if pod:
            ssh = get_ssh_info(pod)
        if ssh:
            break
        time.sleep(15)
        print("  still waiting...")

    if not ssh:
        print("SSH never became available.")
        return

    ip, port = ssh
    print(f"\nSSH available at {ip}:{port}")

    # Test SSH
    for attempt in range(10):
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(port), f"root@{ip}", "echo SSH_OK"],
            capture_output=True, text=True, timeout=15)
        if "SSH_OK" in r.stdout:
            break
        print(f"  SSH not ready yet ({attempt+1}/10)...")
        time.sleep(10)
    else:
        print("SSH refused all attempts.")
        return

    print("SSH confirmed. Running bootstrap...\n")

    # Write bootstrap script to pod then run in nohup
    bootstrap = (
        f"#!/bin/bash\n"
        f"set -x\n"
        f"git clone https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git /workspace/ShinkaEvolveCarePD 2>&1\n"
        f"bash /workspace/ShinkaEvolveCarePD/runpod_startup.sh 2>&1\n"
    )

    # Write the script via SSH heredoc, then run it with nohup
    write_cmd = f"cat > /root/bootstrap.sh << 'ENDOFSCRIPT'\n{bootstrap}\nENDOFSCRIPT\nchmod +x /root/bootstrap.sh"
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-p", str(port), f"root@{ip}", write_cmd],
        text=True, timeout=30)

    # Launch with nohup so it survives disconnect, tee to log
    launch = "nohup bash /root/bootstrap.sh > /root/startup.log 2>&1 &\necho Bootstrap PID: $!"
    r = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-p", str(port), f"root@{ip}", launch],
        capture_output=True, text=True, timeout=15)
    print(r.stdout)

    print("\nBootstrap launched in background.")
    print(f"Monitor with: python runpod_manager.py logs --follow")
    print(f"Or SSH in:    ssh -p {port} root@{ip}")


def cmd_stop():
    pod = get_pod()
    if not pod:
        print("No pod found.")
        return
    pod_id = pod["id"]
    result = gql(f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus }} }}')
    print(f"Stopped: {result}")


def cmd_terminate():
    pod = get_pod()
    if not pod:
        print("No pod found.")
        return
    pod_id = pod["id"]
    ans = input(f"Terminate pod {pod_id} ({pod['name']})? This deletes it. [y/N]: ")
    if ans.lower() != "y":
        print("Cancelled.")
        return
    result = gql(f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
    print(f"Terminated: {result}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RunPod manager for ShinkaEvolve CARE-PD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status",    help="Pod status + balance")

    p_logs = sub.add_parser("logs", help="Get pod logs")
    p_logs.add_argument("--follow", "-f", action="store_true", help="Poll for new logs")
    p_logs.add_argument("--interval", type=int, default=15, help="Poll interval (s)")
    p_logs.add_argument("--tail", type=int, default=100)

    p_watch = sub.add_parser("watch",  help="Live loop: status + leaderboard")
    p_watch.add_argument("--interval", type=int, default=30)

    sub.add_parser("leaderboard", help="Print leaderboard from pod")

    p_deploy = sub.add_parser("deploy", help="Deploy a new pod")
    p_deploy.add_argument("--gpu", default="NVIDIA RTX A5000",
                          help="GPU type ID (default: NVIDIA RTX A5000)")
    p_deploy.add_argument("--count", type=int, default=1)

    sub.add_parser("bootstrap",  help="SSH in and run startup script (after deploy)")
    sub.add_parser("stop",       help="Stop the pod")
    sub.add_parser("terminate",  help="Terminate (delete) the pod")

    p_ssh = sub.add_parser("ssh", help="Run a shell command on the pod")
    p_ssh.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set in .env")
        sys.exit(1)

    if args.cmd == "status":
        cmd_status()
    elif args.cmd == "logs":
        cmd_logs(tail_lines=args.tail, follow=args.follow, interval=args.interval)
    elif args.cmd == "watch":
        cmd_watch(interval=args.interval)
    elif args.cmd == "leaderboard":
        cmd_leaderboard()
    elif args.cmd == "deploy":
        cmd_deploy(gpu_type=args.gpu, gpu_count=args.count)
    elif args.cmd == "bootstrap":
        cmd_bootstrap()
    elif args.cmd == "stop":
        cmd_stop()
    elif args.cmd == "terminate":
        cmd_terminate()
    elif args.cmd == "ssh":
        cmd_ssh(" ".join(args.command))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
