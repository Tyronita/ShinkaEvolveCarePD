# Setup ShinkaEvolve Claude Skill
```bash
npx skills add SakanaAI/ShinkaEvolve --skill '*' -a claude-code -a codex -y
```

# Install Build Tool (uv) and install ShinkaEvolve
```bash
# install Runpod Docs
https://github.com/runpod/docs RunPodDocs

# installs uv
pip install -r requirements.txt

# installs ShinkaEvolve
git clone https://github.com/SakanaAI/ShinkaEvolve
cd ShinkaEvolve
uv venv --python 3.11
.venv\Scripts\activate # source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
cd ..

# uv pip installs
uv pip install datasets
uv pip install python-dotenv

# install Care-PD Repository
git clone https://github.com/TaatiTeam/CARE-PD.git
cd CARE-PD
uv pip install -r requirements.txt
uv pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# set up dataset
mkdir -p assets/datasets
huggingface-cli download vida-adl/CARE-PD --repo-type dataset --local-dir ./assets/datasets
python data/smpl_reader.py --dataset PD-GaM BMCLab 3DGait T-SDU-PD DNE E-LC KUL-DT-T T-LTC T-SDU

# return
cd ..
```

# Secrets Management

## Local Secrets 
Populate .env based on .env.example

### Get your RunPod API Key
https://docs.runpod.io/get-started/api-keys

## RunPod Secrets
Populate https://www.console.runpod.io/user/secrets/ with  same secrets as .env.remote.example.

Accessed like this:
- {{ RUNPOD_SECRET_HF_TOKEN }}

### HF Token
Get your Hugging Face API key - for downloading our dataset
https://huggingface.co/settings

# Dataset Setup

```python
from datasets import load_dataset

# Load CARE-PD dataset from Hugging Face
dataset = load_dataset("vida-adl/CARE-PD")

print(dataset)
```
