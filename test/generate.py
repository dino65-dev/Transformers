import argparse
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Dynamically add repo root to sys.path (works on Colab and locally)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try to import build_transformer from the local package
try:
    from transformer.build_transformer import build_transformer
except Exception as e:
    raise ImportError(
        "Could not import 'build_transformer' from the local 'transformer' package. "
        "Ensure your working directory is the repo root and that the 'transformer' "
        "package (with build_transformer.py) exists.\n"
        f"Repo root detected: {REPO_ROOT}\nOriginal error: {e}"
    )

def load_model_for_inference(checkpoint_path, model, device="cuda"):
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as re:
        # Helpful hint if vocab sizes mismatch (very common if tokenizer differs)
        proj_key = 'projection_layer.proj.weight'
        ckpt_vocab = checkpoint['model_state_dict'].get(proj_key, None)
        model_vocab = dict(model.named_parameters()).get(proj_key, None)
        if ckpt_vocab is not None and model_vocab is not None:
            print(f"State dict mismatch while loading checkpoint.")
            print(f"Checkpoint vocab size: {ckpt_vocab.shape[0]}, Model vocab size: {model_vocab.shape[0]}")
            print("Use the same tokenizer as training (or pass the correct tokenizer/model args).")
        raise re

    model.eval()
    model.to(device)
    print("✅ Model loaded successfully for inference!")
    print(f"   - Trained for {checkpoint.get('epoch', 'N/A')} epochs")
    print(f"   - Final loss: {checkpoint.get('current_loss', float('nan')):.4f}")
    print(f"   - Best loss: {checkpoint.get('best_loss', float('nan')):.4f}")
    return model

def generate(model, tokenizer, prompt, max_new_tokens=100, device="cuda"):
    model.eval()
    device = torch.device(device)
    with torch.no_grad():
        generated_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for _ in range(max_new_tokens):
            seq_len = generated_ids.shape[1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

            # Use the model's API (consistent with training code)
            decoder_out, _ = model.decode(generated_ids, causal_mask)
            logits_last = model.project(decoder_out[:, -1, :])

            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="CLI: Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt) saved by training")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    # Model args (must match training)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--kv_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=2048)
    # Tokenizer source (use the same one you trained with)
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HF model id or local path to tokenizer")
    args = parser.parse_args()

    # Tokenizer setup (match training tokenizer)
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    # Add the same special tokens as training (no-op if already present)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {
        'pad_token': tokenizer.pad_token if tokenizer.pad_token is not None else '[PAD]',
        'additional_special_tokens': ["<user>", "<assistant>"]
    }
    try:
        tokenizer.add_special_tokens(special_tokens)
    except Exception:
        pass

    # Build model with vocab size based on tokenizer to match checkpoint shapes
    device = torch.device(args.device)
    model = build_transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        src_seq_len=args.seq_len,
        tgt_seq_len=args.seq_len,
        d_model=args.d_model,
        N=args.num_layers,
        h=args.num_heads,
        kv_h=args.kv_heads,
        dropout=args.dropout,
        d_ff=args.d_ff
    ).to(device)

    # Do not manually resize/reinit embeddings/projection here; rely on checkpoint weights

    # Load checkpoint
    model = load_model_for_inference(args.checkpoint, model, device=device)

    # Generate
    print("Testing generation:")
    result = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, device=args.device)
    print(f"Generated:\n{result}")

if __name__ == "__main__":
    main()