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
# Import the missing functions/modules
import sys
sys.path.append('/workspaces/Transformers')
from transformer.build_transformer import build_transformer

wandb.login(key="your api key")

def train(model, dataset, tokenizer, device="cuda", epochs=3, batch_size=8, lr=1e-4, 
          checkpoint_dir="checkpoints", use_mixed_precision=True,
          use_wandb=True, project_name="small-model", run_name=None):
    
    # Initialize WandB (unchanged)
    if use_wandb:
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"decoder_training_{timestamp}"
        
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                # Your config here (unchanged)
                "model_type": "decoder_only_transformer",
                "d_model": 768,
                "num_layers": 10,
                "num_heads": 12,
                "num_kv_heads": 4,
                "vocab_size": len(tokenizer),
                "max_sequence_length": 2048,
                "dropout": 0.1,
                "d_ff": 2048,
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
                "architecture_features": ["GQA", "RoPE", "RMSNorm"],
            },
            tags=["decoder-only", "conversational-ai", "gqa", "rope"]
        )
        
        wandb.watch(model, log="all", log_freq=200)
    
    # Rest of the initialization (unchanged)
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
                # ✅ Call without WandB parameter
                save_checkpoint(
                    model, optimizer, epoch, global_step, avg_loss, best_loss,
                    checkpoint_dir, f"auto_checkpoint_epoch_{epoch+1}_step_{global_step}.pt"
                )
                last_checkpoint_time = current_time
                print(f"✅ Checkpoint saved successfully!\n")
            
            # Training step (unchanged)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            if use_mixed_precision:
                with autocast(device_type=device.type):
                    embeddings = model.tgt_embed(input_ids)
                    seq_len = input_ids.size(1)
                    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
                    
                    output = embeddings
                    for layer in model.decoder.layers:
                        output = layer(output, causal_mask, use_cache=False)
                    
                    logits = model.projection_layer(output)
                    
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
                    gpu_memory_gb = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_reserved_gb = torch.cuda.memory_reserved() / 1e9
                    log_dict.update({
                        "system/gpu_memory_allocated_gb": gpu_memory_gb,
                        "system/gpu_memory_reserved_gb": gpu_memory_reserved_gb,
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
            # ✅ Call without WandB parameter
            save_checkpoint(
                model, optimizer, epoch, global_step, avg_loss, best_loss,
                checkpoint_dir, "best_model.pt"
            )
            if use_wandb:
                wandb.log({"train/best_loss": best_loss}, step=global_step)
        
        # ✅ Call without WandB parameter
        save_checkpoint(
            model, optimizer, epoch, global_step, avg_loss, best_loss,
            checkpoint_dir, f"epoch_{epoch+1}_checkpoint.pt"
        )
    
    print("🏁 Training completed! Saving final checkpoint...")
    # ✅ Call without WandB parameter
    save_checkpoint(
        model, optimizer, epochs-1, global_step, avg_loss, best_loss,
        checkpoint_dir, "final_model.pt"
    )
    
    if use_wandb:
        wandb.finish()
    
    return model


def main():
    """Main function for CLI-based training"""
    parser = argparse.ArgumentParser(description="Train a transformer model for conversational AI")
    
    # Model configuration arguments
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=10, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--kv_heads", type=int, default=4, help="Number of key-value heads (for GQA)")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seq_len", type=int, default=2048, help="Maximum sequence length")
    
    # Training configuration arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--checkpoint_dir", type=str, default="./my_model_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_path", type=str, default="trained_transformer.pt", help="Path to save final model")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="lmsys/lmsys-chat-1m", 
                        help="Dataset name or path (compatible with HuggingFace datasets)")
    parser.add_argument("--dataset_config", type=str, default=None, 
                        help="Dataset configuration name")
    parser.add_argument("--dataset_split", type=str, default="train", 
                        help="Dataset split to use")
    parser.add_argument("--dataset_subset", type=int, default=777453, 
                        help="Number of examples to use from dataset (default: 777453)")
    
    # WandB configuration
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--project_name", type=str, default="small-model", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    # Import the tokenizer - assuming it's in the same directory
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
        d_ff=args.d_ff
    )
    
    # Move to device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Resize token embeddings
    print("Resizing token embeddings...")
    model.tgt_embed.weight = torch.nn.Parameter(
        torch.randn(len(tokenizer), args.d_model).to(device)
    )
    
    # Correct weight dimensions
    model.projection_layer.proj.weight = torch.nn.Parameter(
        torch.randn(len(tokenizer), args.d_model).to(device)
    )
    model.projection_layer.proj.bias = torch.nn.Parameter(
        torch.zeros(len(tokenizer)).to(device)
    )
    
    # Apply proper initialization
    torch.nn.init.xavier_uniform_(model.tgt_embed.weight)
    torch.nn.init.xavier_uniform_(model.projection_layer.proj.weight)
    torch.nn.init.zeros_(model.projection_layer.proj.bias)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load dataset based on user arguments
    print(f"Loading dataset: {args.dataset_name}")
    try:
        from datasets import load_dataset
        
        # Load dataset with user-provided parameters
        dataset_kwargs = {}
        if args.dataset_config:
            dataset_kwargs['name'] = args.dataset_config
        
        dataset = load_dataset(
            args.dataset_name, 
            **dataset_kwargs,
            split=args.dataset_split
        )
        
        # Subset dataset if specified
        if args.dataset_subset and args.dataset_subset < len(dataset):
            print(f"Using subset of {args.dataset_subset} examples from {len(dataset)} total")
            dataset = dataset.select(range(args.dataset_subset))
        else:
            print(f"Using full dataset with {len(dataset)} examples")
            
        # Convert to dict format with train key for compatibility
        dataset = {"train": dataset}
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have the dataset available and correctly specified.")
        return
    
    # Start training
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
    




    print("Training complete!")    torch.save(trained_model.state_dict(), args.save_path)    print(f"Saving model to {args.save_path}...")    # Save final model
if __name__ == "__main__":
    main()