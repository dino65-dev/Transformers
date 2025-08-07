# Transformers: A Custom Implementation of Decoder-Only Transformer Models

## Overview

This repository contains a custom implementation of decoder-only transformer models, inspired by modern architectures like GPT. It includes comprehensive code for training conversational AI models on datasets like LMSYS-Chat-1M, featuring advanced techniques such as Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), and RMSNorm.

The implementation demonstrates end-to-end training from scratch, including data preparation, model building, and optimization techniques like mixed precision training. With a flexible command-line interface, users can easily configure model architecture, training parameters, and dataset choices without code modifications. This repository is designed for researchers, developers, and practitioners interested in understanding and implementing transformer architectures for conversational AI applications.

## Key Features

- **Decoder-only transformer architecture** - Clean, modular implementation
- **Custom tokenizer support** - Compatible with BPE tokenizers (e.g., 32K vocabulary)
- **Efficient training** - Mixed precision training with gradient clipping
- **Monitoring integration** - WandB support for tracking loss and metrics
- **Flexible configuration** - Command-line interface for model parameters and dataset selection
- **Dataset flexibility** - Support for various HuggingFace datasets with customizable configurations

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ (with CUDA support for GPU training)
- Hugging Face Transformers library
- Datasets library for data loading
- WandB (optional, for experiment tracking)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/dino65-dev/Transformers.git
   cd Transformers
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch transformers datasets wandb tokenizers
   ```

## Usage

### Model Training via Command Line

Train models directly from the command line with extensive configuration options:

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

#### Core Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--d_model` | Model dimension | 768 |
| `--num_layers` | Number of transformer layers | 10 |
| `--num_heads` | Number of attention heads | 12 |
| `--kv_heads` | Number of key-value heads (GQA) | 4 |
| `--dropout` | Dropout rate | 0.1 |
| `--epochs` | Training epochs | 5 |
| `--batch_size` | Batch size | 6 |
| `--lr` | Learning rate | 3e-4 |

#### Dataset Configuration

Choose from any dataset on HuggingFace or use your own:

```bash
# Using EleutherAI/pile dataset
python train/train.py --dataset_name "EleutherAI/pile" --dataset_config "all" --dataset_subset 10000

# Using Alpaca dataset
python train/train.py --dataset_name "tatsu-lab/alpaca" --dataset_split "train" 
```

Dataset parameters:
- `--dataset_name`: Name of dataset on HuggingFace or path to local dataset
- `--dataset_config`: Configuration name for the dataset (if applicable)
- `--dataset_split`: Dataset split to use (default: "train")
- `--dataset_subset`: Number of examples to use (limit dataset size)

#### WandB Integration

Enable or disable WandB logging:

```bash
# Enable WandB with custom project and run name
python train/train.py --project_name "my-transformer" --run_name "experiment-1"

# Disable WandB
python train/train.py --no_wandb
```

### Training Features

- Automatic checkpointing every 2 hours
- Best model saving based on validation loss
- Mixed precision training (disable with `--no_mixed_precision`)
- GPU memory optimization

### Text Generation

After training, use the model for text generation:

```python
# Example usage in script or notebook
from model import generate_with_cache  # Import your generation function

prompt = "Hello, how are you?"
generated = generate_with_cache(model, tokenizer, prompt, max_length=100)
print(generated)
```

## Project Structure

```
.
├── transformer/          # Core model implementation
│   ├── decoder_blocks.py  # Transformer decoder blocks
│   ├── attention.py       # Attention mechanisms
│   └── embeddings.py      # Position and token embeddings
├── model_train.ipynb      # Main training notebook
├── test-ng.ipynb         # Generation and testing notebook
├── train.py              # Training script (if available)
└── README.md             # This file
```

## Training Configuration

**Default Configuration:**
- **Dataset**: LMSYS-Chat-1M or custom conversational data
- **Tokenizer**: Custom BPE with 32K vocabulary
- **Model Architecture**: 
  - `d_model=128` (embedding dimension)
  - `num_layers=3` (transformer layers)
  - `num_heads=2` (attention heads)
- **Optimization**: AdamW optimizer with mixed precision training
- **Monitoring**: WandB integration for real-time metrics

**Performance Metrics:**
Example training progression:
- Epoch 1: Average Loss ~6.0 (starting from ~10.3)
- Subsequent epochs: Convergence to ~4.0-5.0 range

### Training Loss Progression

![Training Loss](Screenshot%202025-08-01%20214849.png)

*Training loss curve showing the model's learning progression over epochs. The graph demonstrates the typical loss reduction pattern during transformer model training.*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement include:

- **Model Performance**: Enhancing generation coherence and quality
- **Evaluation Metrics**: Adding perplexity and other standard metrics
- **Scalability**: Supporting larger model configurations
- **Documentation**: Improving code documentation and examples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by nanoGPT and Hugging Face Transformers
- Built with modern transformer architecture best practices

For questions, feature requests, or collaboration opportunities, please open an issue or contact via GitHub.

---
**Happy training!** 🚀
