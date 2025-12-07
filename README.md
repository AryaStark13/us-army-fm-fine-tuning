# us-army-fm-fine-tuning
This repository contains code and resources for fine-tuning a language model using US Army Field Manuals (FM) data. The fine-tuned model can be used for various applications, including chat interfaces.

## HF Model Collection

### Gemma Models
| Stage | Model ID | Description |
|-------|----------|-------------|
| Base | `google/gemma-2-9b` | Original Gemma base model |
| CPT-base | [`ShethArihant/gemma-7b-us-army-fm-CPT-base`](https://huggingface.co/ShethArihant/gemma-7b-us-army-fm-CPT-base) | Continued pretraining on Army FM corpus |
| CPT-instruct | [`ShethArihant/gemma-7b-us-army-fm-CPT-instruct`](https://huggingface.co/ShethArihant/gemma-7b-us-army-fm-CPT-instruct) | CPT-base + instruction residuals |
| SFT-instruct | [`ShethArihant/gemma-7b-us-army-fm-SFT-instruct`](https://huggingface.co/ShethArihant/gemma-7b-us-army-fm-SFT-instruct) | Supervised fine-tuned on instruction dataset |

### Llama Models
| Stage | Model ID | Description |
|-------|----------|-------------|
| Base | `meta-llama/Llama-3.1-8B` | Original Llama 3.1 base model |
| CPT-base | [`ShethArihant/Llama-3.1-8B-us-army-fm-CPT-base`](https://huggingface.co/ShethArihant/Llama-3.1-8B-us-army-fm-CPT-base) | Continued pretraining on Army FM corpus |
| CPT-instruct | [`ShethArihant/Llama-3.1-8B-us-army-fm-CPT-instruct`](https://huggingface.co/ShethArihant/Llama-3.1-8B-us-army-fm-CPT-instruct) | CPT-base + instruction residuals |
| SFT-instruct | [`ShethArihant/Llama-3.1-8B-us-army-fm-SFT-instruct`](https://huggingface.co/ShethArihant/Llama-3.1-8B-us-army-fm-SFT-instruct) | Supervised fine-tuned on instruction dataset |

## Setup Instructions (For Local Development)

1. Clone the repo:
```bash
git clone https://github.com/AryaStark13/us-army-fm-fine-tuning.git
```

2. Create and activate env:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install deps:
```bash
pip install -r requirements.txt
# Note: Unsloth does not support M-series Macs. If you're using an M1/M2 Mac,
# please comment out the unsloth line in requirements.txt before installing.
```

4. Copy & fill .evn file:
```bash
cp .env.example .env
```

## Run Streamlit Script
```bash
streamlit run chat_interface.py
```