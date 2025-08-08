# Transformers: A Custom Implementation of Decoder-Only Transformer Models

## Overview
Custom decoder-only Transformer (GPT-style) with Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), and RMSNorm. Train and test from the command line with flexible configuration.

## Key Features
- Decoder-only, modular architecture
- Mixed precision training + gradient clipping
- WandB integration (optional)
- CLI configuration for model, training, and datasets
- Simple CLI text generation for quick testing

## Installation
### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (with CUDA support for GPU training)
- transformers, datasets, tokenizers, wandb (optional)

### Setup
1) Clone and enter repo
```bash
git clone https://github.com/dino65-dev/Transformers.git
cd Transformers
```
2) Virtual environment
```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```
3) Install deps
```bash
pip install torch transformers datasets wandb tokenizers
```

## Quickstart

### 1) Train from CLI
```bash
python train/train.py \
  --d_model 768 \
  --num_layers 10 \
  --num_heads 12 \
  --kv_heads 4 \
  --d_ff 2048 \
  --dropout 0.1 \
  --seq_len 2048 \
  --epochs 5 \
  --batch_size 6 \
  --lr 3e-4 \
  --checkpoint_dir "./my_model_checkpoints" \
  --dataset_name "lmsys/lmsys-chat-1m" \
  --dataset_split "train" \
  --dataset_subset 10000 \
  --project_name "small-model"
```

Common flags:
- Model: --d_model, --num_layers, --num_heads, --kv_heads, --d_ff, --dropout, --seq_len
- Train: --epochs, --batch_size, --lr, --checkpoint_dir, --save_path
- Dataset: --dataset_name, --dataset_config, --dataset_split, --dataset_subset
- System: --device, --no_mixed_precision, --no_wandb, --project_name, --run_name

Notes:
- Checkpoints saved per epoch and as best_model.pt and final_model.pt.
- Resume is removed; each run starts fresh.

### 2) Generate from CLI (Test the model)
```bash
python test/generate.py \
  --checkpoint ./my_model_checkpoints/best_model.pt \
  --prompt "<user> Hello! How are you?" \
  --max_new_tokens 100 \
  --device cuda
```
If trained with non-default hyperparams, pass them:
```bash
python test/generate.py \
  --checkpoint ./my_model_checkpoints/best_model.pt \
  --prompt "Once upon a time" \
  --d_model 768 --num_layers 10 --num_heads 12 --kv_heads 4 --d_ff 2048 --seq_len 2048
```

Tokenizer for generation:
- GPT-2 tokenizer with added special tokens: [PAD], <user>, <assistant>.

## Dataset Tips
- Use any HuggingFace dataset via --dataset_name; pass --dataset_config if needed.
- Limit size quickly with --dataset_subset (e.g., 10k) for dry runs.
- Ensure your dataset returns conversation turns "role"/"content" similar to LMSYS-Chat-1M.

## Training Loss (example)
![Training Loss Curve](https://github.com/dino65-dev/Transformers/blob/main/Screenshot%202025-08-01%20214849.png)

## Project Structure
```
.
├── transformer/                 # Core model
│   ├── build_transformer.py
│   ├── transformer_.py
│   ├── gqa.py, rms_norm.py, residual_connection.py, ...
├── train/
│   ├── train.py                 # CLI training script
│   ├── dataset_define.py
│   ├── save_checkpoint.py
│   └── tokenizer.py
├── test/
│   └── generate.py              # CLI generation script
├── test-ng.ipynb                # Reference notebook
└── README.md
```

## WandB
Enabled by default; disable with --no_wandb. Set API key via environment or code.

## Hardware & Training Time
- Trained on Ola Krutim AI Pod A100 40GB.
- A full run took approximately 6.5 hours.

## Troubleshooting
- CUDA OOM: reduce --batch_size, use --no_mixed_precision off by default (keep mixed precision), shorten --seq_len or --dataset_subset.
- Slow disk or CPU fallback: ensure --device cuda if GPU is available.
- Tokenizer mismatch: keep training and generation tokenizers consistent with added special tokens.

## License
MIT

--- 
**Happy training!** 🚀
## License

MIT

---
**Happy training!** 🚀
