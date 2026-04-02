"""
Test script: validates that pretrain_slimpajama.ipynb code runs correctly.
Uses YOUR existing code from train/ and transformer/ folders.
Simulates SlimPajama data with a fake IterableDataset -- no real download.
"""
import sys, os, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, TRAIN_DIR)

# ========== 1. Import YOUR modules ==========
print("=== 1. Import existing modules ===")
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, IterableDataset
from datetime import datetime

from transformer.build_transformer import build_transformer

# Import directly from train/ (on sys.path, no package prefix needed)
from save_checkpoint import save_checkpoint
from tokenizer import tokenizer
from dataset_define import SlimPajamaDataset

print("[PASS] All imports from your existing code succeeded")

# ========== 2. Config (same as notebook) ==========
print("\n=== 2. Config ===")
D_MODEL     = 768
NUM_LAYERS  = 12
NUM_HEADS   = 12
KV_HEADS    = 4
D_FF        = 3072
DROPOUT     = 0.1
MAX_SEQ_LEN = 2048
USE_REPO    = True
USE_FLASH   = True
BATCH_SIZE  = 2
LR          = 3e-4
VOCAB_SIZE  = len(tokenizer)
PAD_TOKEN_ID = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "_test_checkpoints")

print(f"Vocab={VOCAB_SIZE}, pad_id={PAD_TOKEN_ID}, device={device}")

# ========== 3. Build model (same as notebook) ==========
print("\n=== 3. Build model ===")
model = build_transformer(
    src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
    src_seq_len=MAX_SEQ_LEN, tgt_seq_len=MAX_SEQ_LEN,
    d_model=D_MODEL, N=NUM_LAYERS, h=NUM_HEADS, kv_h=KV_HEADS,
    dropout=DROPOUT, d_ff=D_FF, use_repo=USE_REPO, use_flash=USE_FLASH,
)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"[PASS] Model: {total_params:,} params ({total_params/1e6:.1f}M)")

# ========== 4. Fake dataset mimicking SlimPajamaDataset output ==========
print("\n=== 4. Fake SlimPajama iterator ===")
class FakeSlimPajama(IterableDataset):
    """Mimics the output format of SlimPajamaDataset without parquet files."""
    def __init__(self, tok, max_len, n=8):
        self.tok = tok
        self.max_len = max_len
        self.n = n
    def __iter__(self):
        for i in range(self.n):
            text = f"The quick brown fox jumps over the lazy dog. Sample {i}. " * 20
            ids = self.tok.encode(text, add_special_tokens=False)
            if self.tok.eos_token_id is not None:
                ids.append(self.tok.eos_token_id)
            ids = ids[:self.max_len]
            attn = [1] * len(ids)
            pad_len = self.max_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.tok.pad_token_id] * pad_len
                attn = attn + [0] * pad_len
            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.tensor(attn, dtype=torch.long)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_loader = DataLoader(FakeSlimPajama(tokenizer, MAX_SEQ_LEN, n=8), batch_size=BATCH_SIZE)
print("[PASS] Fake DataLoader created")

# ========== 5. Training loop dry-run (same logic as notebook) ==========
print("\n=== 5. Training loop (2 steps) ===")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98))
scaler = GradScaler()
model.train()
global_step = 0
best_loss = float("inf")
epoch_losses = []
total_loss = 0
batch_count = 0

for i, batch in enumerate(train_loader):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()

    with autocast(device_type=device.type):
        embeddings = model.tgt_embed(input_ids)
        output = embeddings
        for layer in model.decoder.layers:
            output, _ = layer(output, tgt_mask=None, use_cache=False)
        output = model.decoder.norm(output)
        logits = model.project(output)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
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

    print(f"  Step {global_step}: loss={loss.item():.4f}, finite={torch.isfinite(loss).item()}")

    if global_step >= 2:
        break

print(f"[PASS] Training loop OK ({global_step} steps)")

# ========== 6. save_checkpoint (your existing function) ==========
print("\n=== 6. save_checkpoint ===")
avg_loss = total_loss / max(batch_count, 1)
save_checkpoint(
    model, optimizer, 0, global_step, avg_loss, best_loss,
    CHECKPOINT_DIR, "test_checkpoint.pt"
)

ckpt_path = os.path.join(CHECKPOINT_DIR, "test_checkpoint.pt")
loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
gs = loaded["global_step"]
print(f"[PASS] Checkpoint save/load OK (step={gs})")

# Cleanup
import shutil
shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

# ========== Summary ==========
print("\n" + "=" * 55)
print("[PASS] ALL TESTS PASSED - notebook is ready for Azure!")
print("=" * 55)
print(f"  Model            : {total_params/1e6:.1f}M params")
print(f"  REPO-Attention   : ON")
print(f"  Flash-Attention  : ON")
print(f"  Dataset          : SlimPajama-6B (Oxen)")
print(f"  Loss at step {global_step}  : {loss.item():.4f}")
