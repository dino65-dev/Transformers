import torch
import os
import datetime
def save_checkpoint(model, optimizer, epoch, global_step, current_loss, best_loss,
                   checkpoint_dir, filename):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_loss': current_loss,
        'best_loss': best_loss,
        'timestamp': datetime.now().isoformat(),
        'training_args': {
            'lr': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
        }
    }

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved to: {checkpoint_path}")

    # Clean up old auto-checkpoints (keep only last 3)
    if "auto_checkpoint" in filename:
        cleanup_old_checkpoints(checkpoint_dir, keep_last=3)

def cleanup_old_checkpoints(checkpoint_dir, keep_last=3):
    """Remove old auto-checkpoints, keeping only the most recent ones"""
    auto_checkpoints = []

    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("auto_checkpoint") and filename.endswith(".pt"):
            filepath = os.path.join(checkpoint_dir, filename)
            auto_checkpoints.append((filepath, os.path.getmtime(filepath)))

    # Sort by modification time (newest first)
    auto_checkpoints.sort(key=lambda x: x[1], reverse=True)

    # Remove old checkpoints
    for filepath, _ in auto_checkpoints[keep_last:]:
        try:
            os.remove(filepath)
            print(f"🗑️ Removed old checkpoint: {os.path.basename(filepath)}")
        except OSError:
            pass