import time
import os
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
import argparse

from dataset_define import ConversationDataset
from save_checkpoint import save_checkpoint
import sys
import os as _os
# Add the Transformers root to path (works cross-platform)
sys.path.append(_os.path.join(_os.path.dirname(__file__), '..'))
from transformer.build_transformer import build_transformer

wandb.login(key="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")


def train(model, dataset, tokenizer, device="cuda", epochs=3, batch_size=8, lr=1e-4,
          checkpoint_dir="checkpoints", use_mixed_precision=True,
          use_wandb=True, project_name="Spedrox_llm", run_name=None):

    # Initialize WandB
    if use_wandb:
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"decoder_training_{timestamp}"

        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "model_type": "decoder_only_transformer",
                "d_model": 768,
                "num_layers": 12,
                "num_heads": 12,
                "num_kv_heads": 4,
                "vocab_size": len(tokenizer),
                "max_sequence_length": 2048,
                "dropout": 0.1,
                "d_ff": 3072,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "weight_decay": 0.01,
                "gradient_clipping": 1.0,
                "warmup_steps": 500,
                "dataset": "LMSYS-Chat-1M-English",
                "tokenizer_type": "custom_32k",
                "total_conversations": 777453,
                "mixed_precision": use_mixed_precision,
                "device": str(device),
                "architecture_features": ["GQA", "REPO-Attention", "Flash-Attention", "RMSNorm"],
            },
            tags=["decoder-only", "conversational-ai", "gqa", "repo-attention", "flash-attention"]
        )

        wandb.watch(model, log="all", log_freq=200)

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Preparing training with mixed precision: {use_mixed_precision}")
    train_dataset = ConversationDataset(
        dataset["train"].select(range(777453)),
        tokenizer,
        max_length=2048
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    scaler = GradScaler() if use_mixed_precision else None

    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    model.train()
    last_checkpoint_time = time.time()
    epoch_losses = []

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        epoch_start_time = time.time()
        batch_count = 0

        for i, batch in enumerate(train_loader):
            current_time = time.time()

            if current_time - last_checkpoint_time >= 7200:
                print(f"\n🔄 Auto-saving checkpoint at epoch {epoch+1}, batch {i}...")
                avg_loss = total_loss / max(i, 1)
                save_checkpoint(
                    model, optimizer, epoch, global_step, avg_loss, best_loss,
                    checkpoint_dir, f"auto_checkpoint_epoch_{epoch+1}_step_{global_step}.pt"
                )
                last_checkpoint_time = current_time
                print(f"✅ Checkpoint saved successfully!\n")

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_mixed_precision:
                with autocast(device_type=device.type):
                    # Embed tokens
                    embeddings = model.tgt_embed(input_ids)
                    seq_len = input_ids.size(1)

                    # Run through decoder layers (training: no cache, is_causal handled by Flash-Attention)
                    output = embeddings
                    for layer in model.decoder.layers:
                        # use_cache=False → training mode, is_causal handled inside DecoderBlock
                        output, _ = layer(output, tgt_mask=None, use_cache=False)

                    # Final norm is applied inside decoder.forward, but here we call layers directly
                    output = model.decoder.norm(output)

                    # Project to vocab (raw logits)
                    logits = model.project(output)   # (batch, seq_len, vocab_size)

                    # Causal language modelling loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Non-AMP path
                embeddings = model.tgt_embed(input_ids)
                output = embeddings
                for layer in model.decoder.layers:
                    output, _ = layer(output, tgt_mask=None, use_cache=False)
                output = model.decoder.norm(output)
                logits = model.project(output)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            global_step += 1
            batch_count += 1
            epoch_losses.append(loss.item())

            if use_wandb:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                }
                if torch.cuda.is_available():
                    log_dict.update({
                        "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                        "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    })
                wandb.log(log_dict, step=global_step)

            if i % 5 == 0:
                elapsed_time = time.time() - epoch_start_time
                gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}, "
                      f"Time: {elapsed_time:.1f}s, Step: {global_step}, GPU: {gpu_memory:.1f}GB")

            if i % 10 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Duration: {epoch_duration:.1f}s")

        if use_wandb:
            wandb.log({
                "epoch/avg_loss": avg_loss,
                "epoch/duration_seconds": epoch_duration,
                "epoch/batches_processed": batch_count,
                "epoch/min_loss": min(epoch_losses[-batch_count:]),
                "epoch/max_loss": max(epoch_losses[-batch_count:]),
            }, step=global_step)

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"🎯 New best loss: {best_loss:.4f} - Saving best model...")
            save_checkpoint(
                model, optimizer, epoch, global_step, avg_loss, best_loss,
                checkpoint_dir, "best_model.pt"
            )
            if use_wandb:
                wandb.log({"train/best_loss": best_loss}, step=global_step)

        save_checkpoint(
            model, optimizer, epoch, global_step, avg_loss, best_loss,
            checkpoint_dir, f"epoch_{epoch+1}_checkpoint.pt"
        )

    print("🏁 Training completed! Saving final checkpoint...")
    save_checkpoint(
        model, optimizer, epochs - 1, global_step, avg_loss, best_loss,
        checkpoint_dir, "final_model.pt"
    )

    if use_wandb:
        wandb.finish()

    return model


def main():
    """Main function for CLI-based training"""
    parser = argparse.ArgumentParser(description="Train a transformer model for conversational AI")

    # Model configuration
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--kv_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--no_repo", action="store_true", help="Disable REPO-Attention (use standard RoPE)")
    parser.add_argument("--no_flash", action="store_true", help="Disable Flash-Attention")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./my_model_checkpoints")
    parser.add_argument("--save_path", type=str, default="trained_transformer.pt")

    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="lmsys/lmsys-chat-1m")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_subset", type=int, default=777453)

    # WandB
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--project_name", type=str, default="small-model")
    parser.add_argument("--run_name", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("Loading tokenizer...")
    from tokenizer import tokenizer

    print("Building model...")
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
        d_ff=args.d_ff,
        use_repo=not args.no_repo,
        use_flash=not args.no_flash,
    )

    device = torch.device(args.device)
    print(f"Using device: {device}")
    model = model.to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name}")
    try:
        from datasets import load_dataset
        dataset_kwargs = {}
        if args.dataset_config:
            dataset_kwargs['name'] = args.dataset_config
        dataset = load_dataset(args.dataset_name, **dataset_kwargs, split=args.dataset_split)

        if args.dataset_subset and args.dataset_subset < len(dataset):
            print(f"Using subset of {args.dataset_subset} examples from {len(dataset)} total")
            dataset = dataset.select(range(args.dataset_subset))
        else:
            print(f"Using full dataset with {len(dataset)} examples")

        dataset = {"train": dataset}
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Starting training...")
    trained_model = train(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        use_mixed_precision=not args.no_mixed_precision,
        use_wandb=not args.no_wandb,
        project_name=args.project_name,
        run_name=args.run_name
    )

    # Save final model
    print(f"Saving model to {args.save_path}...")
    torch.save(trained_model.state_dict(), args.save_path)
    print("Training complete!")


if __name__ == "__main__":
    main()