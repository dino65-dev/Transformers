# Transformers: A Custom Implementation of Decoder-Only Transformer Models

Overview :
This repository contains a custom implementation of decoder-only transformer models, inspired by modern architectures like GPT. It includes code for training conversational AI models on datasets like LMSYS-Chat-1M, with features such as Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), and RMSNorm. The project demonstrates end-to-end training from scratch, including data preparation, model building, and optimization techniques like mixed precision.

As a student project built in Kolkata, this repo showcases practical AI development on limited resources. It's designed for educational purposes, experimentation, and as a starting point for building chatbots or language models.

Key features:
- Decoder-only transformer architecture
- Support for custom tokenizers (e.g., 32K vocab BPE)
- Efficient training with mixed precision and gradient clipping
- WandB integration for monitoring loss and metrics

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU training)
- Hugging Face Transformers library
- Datasets library (for data loading)
- WandB (optional, for logging)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/dino65-dev/Transformers.git
   cd Transformers
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install torch transformers datasets wandb tokenizers
   ```

## Usage

### Model Training
The main training script is in `model_train.ipynb` or `test-ng.ipynb`. It handles:
- Loading and preprocessing conversational datasets
- Building the transformer model
- Training with custom hyperparameters

Example command to start training (adapt paths as needed):
```
python train.py --dataset_path /path/to/lmsys-chat-1m --epochs 5 --batch_size 8 --lr 3e-4 --use_wandb True
```

Key arguments:
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size (default: 8; adjust based on GPU memory)
- `--lr`: Learning rate (default: 3e-4)
- `--use_wandb`: Enable WandB logging (requires `wandb login`)

During training:
- Loss is logged every 5 batches
- Checkpoints are saved automatically every 2 hours and at epoch end
- WandB tracks loss curves, GPU usage, and more

### Generation and Testing
After training, use the model for text generation:
```python
# In a script or notebook
from model import generate_with_cache  # Assuming you have this function

prompt = "Hello, how are you?"
generated = generate_with_cache(model, tokenizer, prompt, max_length=100)
print(generated)
```

## Project Structure
- **transformer/**: Core model implementation (decoder blocks, attention, embeddings)
- **model_train.ipynb**: Main notebook for training the model
- **test-ng.ipynb**: Testing and generation notebook
- **train.py**: Script version of the training loop (if available)

## Training Details
- **Dataset**: Supports LMSYS-Chat-1M or custom conversational data
- **Tokenizer**: Custom BPE with 32K vocabulary
- **Model Config**: d_model=128, num_layers=3, num_heads=2 (configurable)
- **Optimization**: AdamW with mixed precision, gradient clipping
- **Monitoring**: WandB for real-time loss graphs and metrics

Example loss progression from a sample run:
![](https://github.com/dino65-dev/Transformers/blob/main/Screenshot%202025-08-01%20214849.png)
- Epoch 1: Avg Loss ~6.0 (starting from ~10.3)
- Further epochs show convergence to ~4.0-5.0

## Contributing
As a student project, contributions are welcome! Fork the repo, make changes, and submit a pull request. Focus areas:
- Improving generation coherence
- Adding evaluation metrics (e.g., perplexity)
- Supporting larger models

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by nanoGPT and Hugging Face Transformers
- Built as a learning project in Kolkata, India

For questions or collaboration, open an issue or contact via GitHub. Happy training! 🚀
