# Transformers: Custom Decoder-Only Transformer Implementation

## Overview

This repository contains a comprehensive from-scratch implementation of decoder-only transformer models, featuring advanced techniques like **Grouped Query Attention (GQA)**, **Rotary Positional Embeddings (RoPE)**, and **RMSNorm**. The implementation demonstrates end-to-end training on conversational datasets like LMSYS-Chat-1M with modern optimization techniques including mixed precision training and WandB monitoring.

Designed for researchers, developers, and practitioners interested in understanding and implementing transformer architectures for conversational AI applications.

## Key Features

- **🏗️ Modular Architecture** - Clean, well-structured transformer implementation
- **⚡ Advanced Attention** - Grouped Query Attention (GQA) for efficient inference
- **🔄 Rotary Embeddings** - RoPE for better positional understanding
- **📊 RMSNorm** - Stable normalization technique
- **🚀 Efficient Training** - Mixed precision training with gradient clipping
- **📈 Monitoring** - WandB integration for experiment tracking
- **⚙️ Flexible CLI** - Command-line interface for easy configuration
- **📦 Dataset Support** - Compatible with HuggingFace datasets

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ (with CUDA support for GPU training)
- Hugging Face Transformers and Datasets libraries
- WandB (optional, for experiment tracking)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dino65-dev/Transformers.git
   cd Transformers
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch transformers datasets wandb tokenizers
   ```

## Usage

### Training

Train models with the comprehensive CLI interface:

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
  --checkpoint_dir "./checkpoints" \
  --dataset_name "lmsys/lmsys-chat-1m" \
  --dataset_subset 10000
```

#### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--d_model` | Model dimension | 768 |
| `--num_layers` | Transformer layers | 10 |
| `--num_heads` | Attention heads | 12 |
| `--kv_heads` | Key-value heads (GQA) | 4 |
| `--d_ff` | Feed-forward dimension | 2048 |
| `--dropout` | Dropout rate | 0.1 |
| `--seq_len` | Sequence length | 2048 |
| `--epochs` | Training epochs | 5 |
| `--batch_size` | Batch size | 6 |
| `--lr` | Learning rate | 3e-4 |

#### Dataset Configuration

```bash
# Using different datasets
python train/train.py --dataset_name "EleutherAI/pile" --dataset_config "all"
python train/train.py --dataset_name "tatsu-lab/alpaca" --dataset_split "train"
```

#### WandB Integration

```bash
# Enable WandB logging
python train/train.py --project_name "my-transformer" --run_name "experiment-1"

# Disable WandB
python train/train.py --no_wandb
```

### Text Generation

Generate text using trained models:

```bash
python test/generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --prompt "<user> Hello! How are you?" \
  --max_new_tokens 100 \
  --device cuda
```

For models with custom hyperparameters:

```bash
python test/generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --prompt "Once upon a time" \
  --d_model 768 --num_layers 10 --num_heads 12 --kv_heads 4
```

## Project Structure

```
.
├── transformer/              # Core model implementation
│   ├── build_transformer.py   # Model builder and configuration
│   ├── transformer_.py        # Main transformer model
│   ├── decoder.py             # Decoder wrapper
│   ├── decoder_block.py       # Transformer decoder blocks
│   ├── gqa.py                 # Grouped Query Attention implementation
│   ├── rope.py                # Rotary Position Embeddings
│   ├── rope_helper.py         # RoPE utility functions
│   ├── rms_norm.py            # RMSNorm implementation
│   ├── ff_block.py            # Feed-forward blocks
│   ├── residual_connection.py # Residual connections
│   ├── input_embeddings.py    # Token embeddings
│   ├── positional_encoding.py # Positional encodings
│   └── projection_layer.py    # Output projection
├── train/                     # Training utilities
│   ├── train.py              # Main training script
│   ├── dataset_define.py     # Dataset processing
│   ├── tokenizer.py          # Tokenizer setup
│   ├── coustom_tokenizer.py  # Custom tokenizer implementation
│   └── save_checkpoint.py    # Checkpoint management
├── test/                     # Inference and testing
│   └── generate.py          # Text generation script
├── model_train.ipynb         # Training notebook
├── test-ng.ipynb           # Testing and generation notebook
├── auto_checkpoint_epoch_1_step_48403.pt  # Sample checkpoint
└── Screenshot 2025-08-01 214849.png      # Training loss visualization
```

## Model Architecture

### Core Components

- **Grouped Query Attention (GQA)**: Efficient attention mechanism reducing memory usage
- **Rotary Position Embeddings (RoPE)**: Superior positional encoding for long sequences  
- **RMSNorm**: Stable and efficient normalization
- **Feed-Forward Networks**: SwiGLU activation with configurable dimensions
- **Residual Connections**: Skip connections for stable training

### Default Configuration

- **Model Dimensions**: `d_model=768`, `d_ff=2048`
- **Architecture**: `num_layers=10`, `num_heads=12`, `kv_heads=4`
- **Context Length**: `seq_len=2048`
- **Tokenizer**: GPT-2 tokenizer with special tokens: `[PAD]`, `<user>`, `<assistant>`
- **Optimization**: AdamW with mixed precision training

## Training Results

### Hardware & Performance
- **Hardware**: Ola Krutim AI Pod A100 40GB
- **Training Time**: ~6.5 hours for full run
- **Dataset**: LMSYS-Chat-1M (subset)

### Loss Progression

![Training Loss](Screenshot%202025-08-01%20214849.png)

*Training loss curve showing convergence from ~10.3 to ~4.0-5.0 range over epochs*

**Typical Training Progression:**
- **Epoch 1**: Average Loss ~6.0 (starting from ~10.3)
- **Later Epochs**: Convergence to ~4.0-5.0 range
- **Monitoring**: Real-time tracking via WandB

## Advanced Features

### Training Features
- ✅ Automatic checkpointing every 2 hours
- ✅ Best model saving based on validation loss  
- ✅ Mixed precision training (FP16)
- ✅ Gradient clipping for stability
- ✅ GPU memory optimization
- ✅ Flexible dataset loading from HuggingFace

### Generation Features
- ✅ Configurable sampling parameters
- ✅ Custom prompt templates
- ✅ Efficient inference with KV caching
- ✅ Multi-turn conversation support

## Contributing

Contributions welcome! Areas for enhancement:

- 🎯 **Model Performance**: Improve generation quality and coherence
- 📊 **Evaluation**: Add perplexity and other standard metrics  
- 🔧 **Scalability**: Support for larger model configurations
- 📚 **Documentation**: Enhanced code documentation and examples
- ⚡ **Optimization**: Further training and inference speedups

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by modern transformer architectures and best practices
- Built with PyTorch and Hugging Face ecosystem
- Training infrastructure powered by Ola Krutim AI Pod

---

**Ready to train your own transformer? 🚀**

For questions, feature requests, or collaboration opportunities, please open an issue!