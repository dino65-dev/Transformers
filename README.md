# SPEDROX LLM - 154M Pre-Trained Transformer

A from-scratch implementation of a 154M parameter Decoder-only Transformer, trained on the SlimPajama-6B dataset.

## Architecture Highlights
- **154M Parameters** (Comparable to GPT-2 Small)
- **Decoder-Only** causal language model
- **Grouped Query Attention (GQA)** for faster inference memory
- **Flash-Attention** built-in for rapid training
- **RePo Attention** (Learned Continuous Positions)
- **RMSNorm** and SwiGLU feed-forward network
- **Dataset Streaming**: Uses an `IterableDataset` with PyArrow to stream SlimPajama directly from parquet files, utilizing a **Chunk-Level Shuffle Buffer** to decorrelate long documents and ensure healthy gradients.

---

## 🚀 How to Train

1. Install dependencies:
   ```bash
   pip install torch transformers wandb pyarrow oxen
   ```

2. Download Dataset:
   ```bash
   oxen clone https://hub.oxen.ai/datasets/SlimPajama-6B
   ```

3. **Run Training**:
   Open `pretrain_slimpajama.ipynb` and run the cells.
   The notebook is optimized for **A100 (80GB)** GPUs using `BATCH_SIZE=24` and `GRAD_ACCUM=32` (approx 1.5M tokens per step).

### Resuming Training from Checkpoints
Training auto-saves every 2 hours to the `checkpoints/` folder as `.pt` files.
To resume:
1. Ensure your latest `.pt` file is inside `checkpoints/` (e.g., `auto_epoch1_step5404.pt`).
2. Run the notebook from top to bottom.
3. The **Resume Cell** will automatically detect the file, load the model weights and AdamW optimizer states, and smoothly resume the Cosine Learning Rate Schedule where it left off.

---

## ⚡ Inference & Generation

The repository includes a fast `inference.py` script featuring KV-caching, top-k/top-p sampling, temperature scaling, and repetition penalties.

Run the interactive chat mode:
```bash
python inference.py --checkpoint checkpoints/auto_epoch1_step5404.pt
```

Run demo mode:
```bash
python inference.py --checkpoint checkpoints/auto_epoch1_step5404.pt --demo
```

---

## 📊 Mid-Training Results (Step 5,400)

At ~5,400 steps (about 3 Billion tokens processed), the model reached a loss of **~2.5** (Perplexity ~12). 

### Sample Outputs
Here is what the model generates after partial training:

**Prompt:** `"Once upon a time, in a land far away,"`
> **Output:** "Once upon a time, in a land far away, the first I moved on a year after all you have passed, the last week after you've come to be released by now, though you're back then you were my mind-in' on Earth's it was once again..."
> *(Note: The model demonstrates an understanding of English grammar and temporal reasoning!)*

**Prompt:** `"In 2024, artificial intelligence"`
> **Output:** "In 2024, artificial intelligence has the past 10 years of this event that time we were a recent decades after the summer 2016, we are our first blush of a case you one of a recent years past years, the past my last year 2000..."
> *(Note: The model correctly associates 'artificial intelligence' and '2024' with years and timeframes, though factual hallucination is high at this early stage of training.)*

### Training Health
- **Gradient Norms**: Stabilized at ~0.2 with a peak learning rate of `6e-4`.
- **Loss**: Cleanly broke through the 3.5 plateau into the 2.5 range after applying a Cosine Warm Restart.