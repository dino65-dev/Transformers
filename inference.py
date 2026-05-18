"""
Inference script for the pre-trained decoder-only Transformer.
Supports: top-k, top-p (nucleus), temperature, repetition penalty, KV caching.
"""

import os, sys, torch, argparse
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "train"))

from transformer.build_transformer import build_transformer
from tokenizer import tokenizer

# Suppress HuggingFace warning
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
tokenizer.model_max_length = 2048


# ============================================================
# Model config (must match training)
# ============================================================
MODEL_CONFIG = dict(
    src_vocab_size=len(tokenizer),
    tgt_vocab_size=len(tokenizer),
    src_seq_len=2048,
    tgt_seq_len=2048,
    d_model=768,
    N=12,
    h=12,
    kv_h=4,
    dropout=0.0,   # No dropout at inference
    d_ff=3072,
    use_repo=True,
    use_flash=True,
)


def load_model(checkpoint_path, device="cuda"):
    """Build model architecture and load trained weights."""
    print(f"Building model...")
    model = build_transformer(**MODEL_CONFIG)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The checkpoint may have model_state_dict or be the state_dict directly
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("global_step", "?")
        print(f"  Loaded from step {step}")
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Device: {device}")
    return model


@torch.no_grad()
def generate(
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = "cuda",
):
    """
    Autoregressive text generation with top-k, top-p, temperature,
    and repetition penalty. Uses KV cache for fast generation.
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) == 0:
        input_ids = [tokenizer.bos_token_id or tokenizer.eos_token_id]

    generated = list(input_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # First pass: process the entire prompt, build KV cache
    x = model.tgt_embed(input_tensor)
    layer_caches = None
    x, layer_caches = model.decoder(x, tgt_mask=None, layer_caches=None, use_cache=True)
    logits = model.project(x)

    # Get the last token's logits
    next_logits = logits[:, -1, :]

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    for step in range(max_new_tokens):
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated):
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

        # Temperature scaling
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # Greedy if temperature is 0
            next_token = next_logits.argmax(dim=-1).item()
            generated.append(next_token)
            if next_token == eos_id:
                break
            # Prepare next input
            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            x = model.tgt_embed(input_tensor)
            x, layer_caches = model.decoder(x, tgt_mask=None, layer_caches=layer_caches, use_cache=True)
            next_logits = model.project(x)[:, -1, :]
            continue

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            min_topk = topk_vals[:, -1].unsqueeze(-1)
            next_logits = torch.where(
                next_logits < min_topk,
                torch.full_like(next_logits, float("-inf")),
                next_logits,
            )

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            # Scatter back
            next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        # Stop on EOS or PAD
        if next_token == eos_id or next_token == pad_id:
            break

        # Prepare next step with KV cache (only process the new token)
        input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        x = model.tgt_embed(input_tensor)
        x, layer_caches = model.decoder(x, tgt_mask=None, layer_caches=layer_caches, use_cache=True)
        next_logits = model.project(x)[:, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True)


def interactive_mode(model, device):
    """Interactive chat-like interface."""
    print("\n" + "=" * 60)
    print("   SPEDROX LLM - Interactive Inference")
    print("   Model: 154M Decoder-Only Transformer")
    print("   Trained on: SlimPajama-6B (~5.4k steps)")
    print("=" * 60)
    print("\nCommands:")
    print("  /quit        - Exit")
    print("  /temp 0.5    - Set temperature (default: 0.8)")
    print("  /topk 40     - Set top-k (default: 50)")
    print("  /topp 0.85   - Set top-p (default: 0.9)")
    print("  /tokens 512  - Set max new tokens (default: 256)")
    print("  /rep 1.3     - Set repetition penalty (default: 1.2)")
    print("  /greedy      - Switch to greedy decoding")
    print()

    settings = {
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "max_new_tokens": 256,
        "repetition_penalty": 1.2,
    }

    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        # Handle commands
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()
            if cmd == "/quit":
                print("Goodbye!")
                break
            elif cmd == "/temp" and len(parts) > 1:
                settings["temperature"] = float(parts[1])
                print(f"  Temperature set to {settings['temperature']}")
            elif cmd == "/topk" and len(parts) > 1:
                settings["top_k"] = int(parts[1])
                print(f"  Top-k set to {settings['top_k']}")
            elif cmd == "/topp" and len(parts) > 1:
                settings["top_p"] = float(parts[1])
                print(f"  Top-p set to {settings['top_p']}")
            elif cmd == "/tokens" and len(parts) > 1:
                settings["max_new_tokens"] = int(parts[1])
                print(f"  Max tokens set to {settings['max_new_tokens']}")
            elif cmd == "/rep" and len(parts) > 1:
                settings["repetition_penalty"] = float(parts[1])
                print(f"  Repetition penalty set to {settings['repetition_penalty']}")
            elif cmd == "/greedy":
                settings["temperature"] = 0
                print("  Switched to greedy decoding")
            else:
                print(f"  Unknown command: {cmd}")
            continue

        # Generate
        print("\n--- Generating ---")
        output = generate(model, prompt, device=device, **settings)

        # Show the generated part (after the prompt)
        print(f"\n{output}")
        print("--- Done ---")


def demo_prompts(model, device):
    """Run a few demo prompts to showcase the model."""
    prompts = [
        "The meaning of life is",
        "In a world where artificial intelligence",
        "The quick brown fox jumped over the",
        "Once upon a time, in a land far away,",
        "Python is a programming language that",
        "The capital of France is",
        "Scientists have recently discovered that",
    ]

    print("\n" + "=" * 60)
    print("   DEMO: Running sample prompts")
    print("=" * 60)

    for i, prompt in enumerate(prompts):
        print(f"\n{'-' * 50}")
        print(f"Prompt {i+1}: \"{prompt}\"")
        print(f"{'-' * 50}")

        output = generate(
            model, prompt, device=device,
            max_new_tokens=128, temperature=0.7,
            top_k=50, top_p=0.9, repetition_penalty=1.2,
        )
        print(output)

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for pre-trained Transformer")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="auto_epoch1_step5404.pt",
                        help="Path to checkpoint .pt file")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="Device (cuda/cpu, auto-detected if not set)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo prompts instead of interactive mode")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Single prompt (non-interactive)")
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--max-tokens", "-n", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    model = load_model(args.checkpoint, device=device)

    if args.prompt:
        # Single prompt mode
        output = generate(
            model, args.prompt, device=device,
            max_new_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
        )
        print(output)
    elif args.demo:
        demo_prompts(model, device)
    else:
        interactive_mode(model, device)
