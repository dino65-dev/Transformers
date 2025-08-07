# Transformers: A Custom Implementation of Decoder-Only Transformer Models

## Overview

This repository contains a custom implementation of decoder-only transformer models, inspired by modern architectures like GPT. It includes comprehensive code for training conversational AI models on datasets like LMSYS-Chat-1M, featuring advanced techniques such as Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), and RMSNorm.

The implementation demonstrates end-to-end training from scratch, including data preparation, model building, and optimization techniques like mixed precision training. This repository is designed for researchers, developers, and practitioners interested in understanding and implementing transformer architectures for conversational AI applications.

## Key Features

- **Decoder-only transformer architecture** - Clean, modular implementation
- **Custom tokenizer support** - Compatible with BPE tokenizers (e.g., 32K vocabulary)
- **Efficient training** - Mixed precision training with gradient clipping
- **Monitoring integration** - WandB support for tracking loss and metrics
- **Flexible configuration** - Easily customizable model parameters

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

### Model Training

The main training implementation is available in `model_train.ipynb` and `test-ng.ipynb`. The training pipeline handles:

- Loading and preprocessing conversational datasets
- Building the transformer model architecture
- Training with customizable hyperparameters
- Automatic checkpointing and logging

Example training command (adapt paths as needed):
```bash
python train.py --dataset_path /path/to/lmsys-chat-1m --epochs 5 --batch_size 8 --lr 3e-4 --use_wandb True
```

**Key Arguments:**
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size (default: 8; adjust based on GPU memory)
- `--lr`: Learning rate (default: 3e-4)
- `--use_wandb`: Enable WandB logging (requires `wandb login`)

**Training Features:**
- Loss logging every 5 batches
- Automatic checkpointing every 2 hours and at epoch completion
- Real-time monitoring via WandB (loss curves, GPU utilization, etc.)

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
