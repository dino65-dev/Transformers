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

def create_32k_tokenizer(dataset_name="lmsys/lmsys-chat-1m", vocab_size=32000):
    """Create a custom 32K BPE tokenizer - identical to the notebook implementation"""
    print(f"Creating 32K tokenizer from {dataset_name}...")
    
    # Import necessary libraries
    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
    except ImportError as e:
        raise ImportError(f"Missing required packages for tokenizer creation: {e}")

    # Load the dataset
    dataset = load_dataset(dataset_name)
    dataset = dataset.filter(lambda x: x['language'] == 'English')
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Setup trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    
    # Prepare training data
    def get_training_corpus():
        for item in dataset["train"]:
            conversation = item['conversation']
            for turn in conversation:
                yield turn['content']
    
    # Train tokenizer
    print("Training tokenizer from dataset... (this may take a while)")
    tokenizer.train_from_iterator(get_training_corpus(), trainer)
    
    # Convert to HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.pad_token = "<pad>"
    hf_tokenizer.eos_token = "<eos>"
    hf_tokenizer.bos_token = "<bos>"
    hf_tokenizer.unk_token = "<unk>"
    
    # Add the same special tokens as in training
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    special_tokens = {
        'pad_token': '[PAD]',
        'additional_special_tokens': ["<user>", "<assistant>"]
    }
    num_added = hf_tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens to match training")
    
    print(f"✅ 32K tokenizer created with vocab size: {len(hf_tokenizer)}")
    return hf_tokenizer

def load_model_for_inference(checkpoint_path, model, device="cuda"):
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        # Tolerate extra keys from training (e.g., tgt_embed.weight) and report
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            print(f"⚠️ Missing keys: {len(missing)} (showing first 5) -> {missing[:5]}")
        if unexpected:
            print(f"⚠️ Unexpected keys: {len(unexpected)} (showing first 5) -> {unexpected[:5]}")
    except RuntimeError as re:
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
    # Tokenizer options
    parser.add_argument("--tokenizer", type=str, default=None, 
                      help="Path to saved tokenizer directory (if omitted, creates 32K tokenizer)")
    parser.add_argument("--save_tokenizer", type=str, default=None, 
                      help="Directory to save the 32K tokenizer for future use")
    parser.add_argument("--dataset", type=str, default="lmsys/lmsys-chat-1m", 
                      help="Dataset to use for creating 32K tokenizer")
    args = parser.parse_args()

    # Tokenizer setup - use the same 32K tokenizer from notebook by default
    if args.tokenizer:
        # Load from saved tokenizer if provided
        from transformers import AutoTokenizer
        print(f"Loading tokenizer from {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        # Create 32K tokenizer matching the notebook
        tokenizer = create_32k_tokenizer(args.dataset, vocab_size=32000)
        
        # Save tokenizer if requested
        if args.save_tokenizer:
            save_dir = args.save_tokenizer
            os.makedirs(save_dir, exist_ok=True)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved 32K tokenizer to {save_dir}")

    # Build model with tokenizer vocab size for exact match
    device = torch.device(args.device)
    vocab_size = len(tokenizer)
    print(f"Building model with vocabulary size: {vocab_size}")
    
    model = build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=args.seq_len,
        tgt_seq_len=args.seq_len,
        d_model=args.d_model,
        N=args.num_layers,
        h=args.num_heads,
        kv_h=args.kv_heads,
        dropout=args.dropout,
        d_ff=args.d_ff
    ).to(device)

    # Load checkpoint
    model = load_model_for_inference(args.checkpoint, model, device=device)

    # Generate
    print("Testing generation:")
    result = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, device=args.device)
    print(f"Generated:\n{result}")

if __name__ == "__main__":
    main()