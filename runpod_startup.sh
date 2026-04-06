#!/usr/bin/env bash
# =============================================================================
# runpod_startup.sh — CARE-PD ShinkaEvolve RunPod startup script
#
# This script is designed to run as a RunPod pod startup command.
# It handles: environment setup, dependency installation, pipeline validation,
# ShinkaEvolve docs hosting, Web UI, evolution run, and git-based persistence.
#
# Usage (RunPod pod startup command field):
#   bash /workspace/ShinkaEvolveCarePD/runpod_startup.sh
#
# Optional env overrides (set as pod environment variables):
#   NUM_GENERATIONS=100      (default: 100)
#   MAX_EVAL_JOBS=4          (default: 4)
#   MAX_API_COSTS=50.0       (default: 50.0)
#   SMOKE_ONLY=true          (default: false — run smoke test only, no full run)
#   SKIP_DOCS=false          (default: false — serve ShinkaEvolve docs)
#   RESULTS_DIR=/workspace/ShinkaEvolveCarePD/results/care_pd_full
#
# Required RunPod secrets (mapped as pod environment variables):
#   ANTHROPIC_API_KEY    {{ RUNPOD_SECRET_ANTHROPIC_API_KEY }}
#   GEMINI_API_KEY       {{ RUNPOD_SECRET_GEMINI_API_KEY }}
#   HF_TOKEN             {{ RUNPOD_SECRET_HF_TOKEN }}
#   GITHUB_TOKEN         {{ RUNPOD_SECRET_GITHUB_TOKEN }}
#   GITHUB_REPO          {{ RUNPOD_SECRET_GITHUB_REPO }}   (e.g. YourUser/ShinkaEvolveCarePD)
# =============================================================================

set -euo pipefail

# ── 0. SSH setup (we bypass /start.sh via dockerArgs, so do it ourselves) ─────
# RunPod injects the user's public key as $PUBLIC_KEY env var.
mkdir -p /root/.ssh && chmod 700 /root/.ssh
[ -n "${PUBLIC_KEY:-}" ] && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A >/dev/null 2>&1 || true   # generate host keys if missing
if [ -f /usr/sbin/sshd ]; then
  /usr/sbin/sshd &
  echo "[startup] sshd started (PID $!)"
fi

# ── Map RUNPOD_SECRET_* → plain env vars (fallback if UI mapping wasn't set) ─
# RunPod injects secrets as RUNPOD_SECRET_<NAME>. The UI lets you map them to
# plain names, but if that step was skipped this block handles it automatically.
: "${ANTHROPIC_API_KEY:=${RUNPOD_SECRET_ANTHROPIC_API_KEY:-}}"
: "${GEMINI_API_KEY:=${RUNPOD_SECRET_GEMINI_API_KEY:-}}"      # embeddings only
: "${OPENROUTER_API_KEY:=${RUNPOD_SECRET_OPENROUTER_API_KEY:-}}"
: "${AZURE_API_KEY:=${RUNPOD_SECRET_AZURE_API_KEY:-}}"
: "${HF_TOKEN:=${RUNPOD_SECRET_HF_TOKEN:-}}"
: "${GITHUB_TOKEN:=${RUNPOD_SECRET_GITHUB_TOKEN:-}}"
: "${GITHUB_EMAIL:=${RUNPOD_SECRET_GITHUB_EMAIL:-}}"
# Plain pod env vars (not secrets):
# GITHUB_REPO, AZURE_API_ENDPOINT, AZURE_API_VERSION, NUM_GENERATIONS, etc.

# ── Config from env (with defaults) ─────────────────────────────────────────
WORKSPACE="${WORKSPACE:-/workspace}"
NUM_GENERATIONS="${NUM_GENERATIONS:-100}"
MAX_EVAL_JOBS="${MAX_EVAL_JOBS:-4}"
MAX_PROPOSAL_JOBS="${MAX_PROPOSAL_JOBS:-4}"
MAX_API_COSTS="${MAX_API_COSTS:-50.0}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"
SKIP_DOCS="${SKIP_DOCS:-false}"
DOCS_PORT="${DOCS_PORT:-8888}"
WEBUI_PORT="${WEBUI_PORT:-8080}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE/ShinkaEvolveCarePD/results/care_pd_full}"

SHINKA_DIR="$WORKSPACE/ShinkaEvolve"
CARE_PD_DIR="$WORKSPACE/CARE-PD"
TASK_DIR="$WORKSPACE/ShinkaEvolveCarePD"

log() { echo "[startup] $(date -u '+%H:%M:%S') $*"; }

# ── 1. Clone / update repos ──────────────────────────────────────────────────
log "=== Step 1: Repositories ==="

if [ -z "${GITHUB_REPO:-}" ]; then
  log "WARNING: GITHUB_REPO not set — skipping git clone of ShinkaEvolveCarePD"
  log "         Assuming repo already at $TASK_DIR via network volume"
else
  if [ ! -d "$TASK_DIR/.git" ]; then
    log "Cloning ShinkaEvolveCarePD..."
    git clone "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" "$TASK_DIR"
  else
    log "Pulling latest ShinkaEvolveCarePD..."
    cd "$TASK_DIR" && git pull --ff-only 2>/dev/null || log "WARNING: git pull failed, using existing code"
  fi
fi

if [ ! -d "$SHINKA_DIR/.git" ]; then
  log "Cloning ShinkaEvolve..."
  git clone https://github.com/SakanaAI/ShinkaEvolve.git "$SHINKA_DIR"
fi

if [ ! -d "$CARE_PD_DIR/.git" ]; then
  log "Cloning CARE-PD..."
  git clone https://github.com/TaatiTeam/CARE-PD.git "$CARE_PD_DIR"
fi

# ── 2. Python environment ────────────────────────────────────────────────────
log "=== Step 2: Python environment ==="

pip install uv -q 2>/dev/null || true

cd "$SHINKA_DIR"
if [ ! -d ".venv" ]; then
  log "Creating venv with uv (inheriting system site-packages for pre-installed torch)..."
  uv venv --python 3.11 --system-site-packages
fi

source .venv/bin/activate
log "Python: $(python --version)"

log "Installing ShinkaEvolve..."
uv pip install -e . -q

log "Installing additional deps..."
uv pip install python-dotenv datasets scikit-learn scipy huggingface_hub smplx -q

# PyTorch is pre-installed in the RunPod image — skip reinstall.
# The venv inherits it via --system-site-packages.
log "Checking inherited PyTorch..."
python -c "import torch; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}')"

# ── 3. Apply sys.executable fix to scheduler.py ──────────────────────────────
log "=== Step 3: Patching scheduler.py ==="

SCHEDULER="$SHINKA_DIR/shinka/launch/scheduler.py"
if grep -q '"python"' "$SCHEDULER" 2>/dev/null; then
  log "Applying sys.executable fix..."
  # Add sys import if missing
  if ! grep -q "^import sys" "$SCHEDULER"; then
    sed -i '1s/^/import sys\n/' "$SCHEDULER"
  fi
  # Replace hardcoded "python" with sys.executable
  sed -i 's/\["python",/[sys.executable,/g' "$SCHEDULER"
  sed -i "s/python_cmd = \[\"python\"\]/python_cmd = [sys.executable]/g" "$SCHEDULER"
  log "scheduler.py patched"
else
  log "scheduler.py already patched (sys.executable found)"
fi

# ── 3b. GPU health check ─────────────────────────────────────────────────────
log "=== Step 3b: GPU health check ==="
python -c "
import torch
cuda_ok = torch.cuda.is_available()
n_gpu   = torch.cuda.device_count() if cuda_ok else 0
name    = torch.cuda.get_device_name(0) if cuda_ok else 'N/A'
mem_gb  = (torch.cuda.get_device_properties(0).total_memory / 1e9) if cuda_ok else 0
print(f'CUDA: {cuda_ok}  GPUs: {n_gpu}  Device: {name}  VRAM: {mem_gb:.1f} GB')
if not cuda_ok:
    print('WARNING: No CUDA GPU detected — evolution will run on CPU (much slower)')
"

# ── 4. Dataset download ──────────────────────────────────────────────────────
log "=== Step 4: Dataset ==="

DATASET_PKL="$CARE_PD_DIR/assets/datasets/BMCLab.pkl"
if [ ! -f "$DATASET_PKL" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    log "ERROR: HF_TOKEN not set and BMCLab.pkl not found. Cannot download dataset."
    exit 1
  fi
  log "Downloading CARE-PD dataset from HuggingFace..."
  mkdir -p "$CARE_PD_DIR/assets/datasets"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vida-adl/CARE-PD',
    repo_type='dataset',
    local_dir='$CARE_PD_DIR/assets/datasets',
    token='$HF_TOKEN',
    ignore_patterns=['*.git*'],
)
print('Download complete')
"
  log "Dataset downloaded"
else
  log "Dataset already present: $DATASET_PKL"
fi

# ── 4b. Pretrained checkpoints (gdown from Google Drive) ─────────────────────
CKPT_DIR="$CARE_PD_DIR/assets/Pretrained_checkpoints"
MOTIONCLIP_CKPT="$CKPT_DIR/motionclip/motionclip_encoder_checkpoint_0100.pth.tar"
if [ ! -f "$MOTIONCLIP_CKPT" ]; then
  log "Downloading pretrained checkpoints (via gdown)..."
  uv pip install gdown -q
  mkdir -p "$CARE_PD_DIR/assets"
  cd "$CARE_PD_DIR/assets"
  gdown 1n-iZFKWmcy6UIQgW9YAkT4Y2DQILSIrp
  unzip -q Pretrained_checkpoints.zip
  rm -f Pretrained_checkpoints.zip
  cd "$SHINKA_DIR"
  log "Checkpoints downloaded"
else
  log "Pretrained checkpoints already present"
fi

# ── 5. Create .env ────────────────────────────────────────────────────────────
log "=== Step 5: Writing .env ==="

cat > "$TASK_DIR/.env" << EOF
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
GEMINI_API_KEY=${GEMINI_API_KEY:-}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
HF_TOKEN=${HF_TOKEN:-}
GITHUB_TOKEN=${GITHUB_TOKEN:-}
EOF
log ".env written to $TASK_DIR/.env"

# ── 6. Validate pipeline (standalone evaluate.py smoke test) ─────────────────
log "=== Step 6: Pipeline validation ==="

VALIDATE_RESULTS="$TASK_DIR/care_pd_task/results_validate"
mkdir -p "$VALIDATE_RESULTS"

log "Running evaluate.py with initial.py..."
python "$TASK_DIR/care_pd_task/evaluate.py" \
  --program_path "$TASK_DIR/care_pd_task/initial.py" \
  --results_dir "$VALIDATE_RESULTS"

if [ -f "$VALIDATE_RESULTS/metrics.json" ]; then
  SCORE=$(python -c "import json; d=json.load(open('$VALIDATE_RESULTS/metrics.json')); print(d.get('combined_score','?'))")
  log "Validation PASSED — baseline macro_f1=$SCORE"
else
  log "ERROR: metrics.json not written — pipeline validation failed"
  exit 1
fi

if [ "$SMOKE_ONLY" = "true" ]; then
  log "SMOKE_ONLY=true — stopping after validation"
  exit 0
fi

# ── 6b. Preflight benchmark — all methods scored before evolution starts ──────
log "=== Step 6b: Preflight benchmark (all methods, fold 0) ==="
log "Testing RandomForest, GradientBoosting, 1D-CNN, MotionCLIP — results below:"
python "$TASK_DIR/care_pd_task/preflight.py" --folds 0
log "=== Preflight complete — starting evolution ==="

# ── 7. Serve ShinkaEvolve docs ───────────────────────────────────────────────
if [ "$SKIP_DOCS" != "true" ] && [ -d "$SHINKA_DIR/docs" ]; then
  log "=== Step 7: ShinkaEvolve docs on port $DOCS_PORT ==="
  cd "$SHINKA_DIR/docs"
  python -m http.server "$DOCS_PORT" --bind 0.0.0.0 &
  DOCS_PID=$!
  log "Docs server PID=$DOCS_PID — http://localhost:$DOCS_PORT"
  cd "$TASK_DIR"
fi

# ── 8. Start shinka_visualize Web UI ─────────────────────────────────────────
log "=== Step 8: ShinkaEvolve Web UI on port $WEBUI_PORT ==="

mkdir -p "$RESULTS_DIR"

# Run in background; SSH tunnel to access: ssh -L 8080:localhost:8080 <pod>
shinka_visualize "$RESULTS_DIR" --port "$WEBUI_PORT" &
WEBUI_PID=$!
log "Web UI PID=$WEBUI_PID — http://localhost:$WEBUI_PORT"
log "  SSH tunnel: ssh -L $WEBUI_PORT:localhost:$WEBUI_PORT root@<pod-ip> -p <port> -i <key>"

# ── 9. Git persistence helper (runs every 30 min in background) ──────────────
log "=== Step 9: Git persistence daemon ==="

if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${GITHUB_REPO:-}" ]; then
  cat > /tmp/git_persist.sh << 'GITSCRIPT'
#!/bin/bash
TASK_DIR="$1"
GITHUB_TOKEN="$2"
GITHUB_REPO="$3"

cd "$TASK_DIR"

# Configure git auth
git config user.email "runpod@shinka.ai" 2>/dev/null || true
git config user.name "RunPod ShinkaEvolve" 2>/dev/null || true
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" 2>/dev/null || true

while true; do
  sleep 1800  # 30 minutes
  echo "[git-persist] $(date -u '+%H:%M:%S') Committing results..."

  # Stage leaderboard + results (avoid large model files)
  git add care_pd_task/leaderboard.csv 2>/dev/null || true
  git add "results/care_pd_full/**/*.json" 2>/dev/null || true
  git add "results/care_pd_full/**/*.py" 2>/dev/null || true
  git add "results/care_pd_full/**/*.yaml" 2>/dev/null || true

  if ! git diff --cached --quiet; then
    git commit -m "RunPod auto-commit: $(date -u '+%Y-%m-%dT%H:%M:%SZ') [skip ci]" 2>/dev/null || true
    git push 2>/dev/null && echo "[git-persist] Pushed successfully" || echo "[git-persist] Push failed (will retry)"
  else
    echo "[git-persist] No changes to commit"
  fi
done
GITSCRIPT
  chmod +x /tmp/git_persist.sh
  bash /tmp/git_persist.sh "$TASK_DIR" "$GITHUB_TOKEN" "$GITHUB_REPO" &
  GIT_PID=$!
  log "Git persistence daemon PID=$GIT_PID — pushes every 30 min"

  # Configure git remote immediately
  cd "$TASK_DIR"
  git config user.email "${GITHUB_EMAIL:-runpod@shinka.ai}" 2>/dev/null || true
  git config user.name "Tyronita" 2>/dev/null || true
  git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" 2>/dev/null || true
  cd "$SHINKA_DIR"
else
  log "GITHUB_TOKEN/GITHUB_REPO not set — git persistence disabled"
  log "  Results will be on network volume only"
fi

# ── 10. Launch ShinkaEvolve ──────────────────────────────────────────────────
log "=== Step 10: ShinkaEvolve Evolution ==="
log "  Generations: $NUM_GENERATIONS"
log "  Max API cost: \$$MAX_API_COSTS"
log "  Eval jobs: $MAX_EVAL_JOBS"
log "  Results: $RESULTS_DIR"
log ""
log "  Web UI:  http://localhost:$WEBUI_PORT"
log "  Docs:    http://localhost:$DOCS_PORT"
log ""

cd "$TASK_DIR"
shinka_run \
  --task-dir care_pd_task \
  --results_dir "$RESULTS_DIR" \
  --num_generations "$NUM_GENERATIONS" \
  --config-fname shinka_task.yaml \
  --set evo.max_api_costs="$MAX_API_COSTS" \
  --max-evaluation-jobs "$MAX_EVAL_JOBS" \
  --max-proposal-jobs "$MAX_PROPOSAL_JOBS"

# ── 11. Final git push after run completes ───────────────────────────────────
log "=== Step 11: Final git push ==="

if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${GITHUB_REPO:-}" ]; then
  cd "$TASK_DIR"
  git add care_pd_task/leaderboard.csv 2>/dev/null || true
  git add --all results/care_pd_full/ 2>/dev/null || true

  if ! git diff --cached --quiet; then
    git commit -m "RunPod completed: $NUM_GENERATIONS gens, cost_cap=\$$MAX_API_COSTS [$(date -u '+%Y-%m-%dT%H:%M:%SZ')]"
    git push
    log "Final push complete"
  else
    log "No new changes to push"
  fi
fi

log "=== Run complete ==="
