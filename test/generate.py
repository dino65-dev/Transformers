import argparse
import os
import sys
import torch
import torch.nn.functional as F

# Ensure local imports work
sys.path.append('/workspaces/Transformers')

from transformer.build_transformer import build_transformer

def load_model_for_inference(checkpoint_path, model, device="cuda"):
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
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
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # 0/1 mask

            # Forward pass like in the notebook training cell
            embeddings = model.tgt_embed(generated_ids)
            output = embeddings
            for layer in model.decoder.layers:
                output = layer(output, causal_mask, use_cache=False)  # returns tensor in training mode

            # Project only the last token
            logits = model.projection_layer(output[:, -1, :])

            # Notebook used softmax for sampling
            probs = F.softmax(logits, dim=-1)
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
    # Model args (same defaults as notebook/train)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--kv_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=2048)
    args = parser.parse_args()

    # Tokenizer setup (mirrors notebook behavior)
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {
        'pad_token': '[PAD]',
        'additional_special_tokens': ["<user>", "<assistant>"]
    }
    tokenizer.add_special_tokens(special_tokens)

    # Build model
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
    )

    # Match notebook resizing/initialization
    device = torch.device(args.device)
    model = model.to(device)
    model.tgt_embed.weight = torch.nn.Parameter(
        torch.randn(len(tokenizer), args.d_model).to(device)
    )
    model.projection_layer.proj.weight = torch.nn.Parameter(
        torch.randn(len(tokenizer), args.d_model).to(device)
    )
    model.projection_layer.proj.bias = torch.nn.Parameter(
        torch.zeros(len(tokenizer)).to(device)
    )
    torch.nn.init.xavier_uniform_(model.tgt_embed.weight)
    torch.nn.init.xavier_uniform_(model.projection_layer.proj.weight)
    torch.nn.init.zeros_(model.projection_layer.proj.bias)

    # Load checkpoint
    model = load_model_for_inference(args.checkpoint, model, device=device)

    # Generate
    print("Testing generation:")
    result = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, device=args.device)
    print(f"Generated:\n{result}")

if __name__ == "__main__":
    main()