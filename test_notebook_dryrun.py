"""Verify the fixed notebook logic: RMSNorm gamma + LR schedule + training."""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train"))

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, IterableDataset
from transformer.build_transformer import build_transformer
from tokenizer import tokenizer

VOCAB_SIZE = len(tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Build model and verify gammas
print("=== 1. Build model ===")
model = build_transformer(VOCAB_SIZE, VOCAB_SIZE, 2048, 2048,
    d_model=768, N=12, h=12, kv_h=4, dropout=0.1, d_ff=3072,
    use_repo=True, use_flash=True)
model = model.to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# CRITICAL: Check RMSNorm gamma is NOT zero
for name, p in model.named_parameters():
    if 'gamma' in name:
        val = p.abs().mean().item()
        status = "[OK]" if val > 0.5 else "[FAIL]"
        print(f"  {status} {name}: mean_abs={val:.4f}")
        assert val > 0.5, f"FATAL: {name} is near zero!"
print("[PASS] All gamma params are non-zero")

# 2. Verify output is not dead
print("\n=== 2. Output check ===")
model.train()
dummy = torch.randint(0, VOCAB_SIZE, (2, 128), device=device)
with torch.no_grad():
    x = model.tgt_embed(dummy)
    for layer in model.decoder.layers:
        x, _ = layer(x, tgt_mask=None, use_cache=False)
    x = model.decoder.norm(x)
    logits = model.project(x)
print(f"Logits std: {logits.std().item():.4f}")
assert logits.std().item() > 0.01, "FATAL: Dead output!"
print("[PASS] Model output is alive")

# 3. LR schedule
print("\n=== 3. LR schedule ===")
LR = 3e-4; MIN_LR = 3e-5; WARMUP = 200; TOTAL = 15258
def get_lr(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = min((step - WARMUP) / max(1, TOTAL - WARMUP), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (LR - MIN_LR) * cosine

for s in [0, 10, 50, 100, 200, 500, 2000, 10000]:
    print(f"  Step {s:>6}: {get_lr(s):.8f}")
assert get_lr(0) > 0, "LR at step 0 must be > 0"
assert get_lr(100) > 1e-4, "LR at step 100 must be meaningful"
assert get_lr(200) > 2.9e-4, "LR at warmup end must be near peak"
print("[PASS] LR schedule looks correct")

# 4. Training steps
print("\n=== 4. Training (3 steps) ===")
class FakeDS(IterableDataset):
    def __init__(self, n=32):
        self.n = n
    def __iter__(self):
        for _ in range(self.n):
            ids = torch.randint(0, VOCAB_SIZE, (2048,))
            yield {"input_ids": ids, "labels": ids.clone()}

dl = DataLoader(FakeDS(32), batch_size=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
for pg in optimizer.param_groups:
    pg['lr'] = get_lr(0)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler = GradScaler(enabled=(not use_bf16))
GRAD_ACCUM = 4
micro = 0
step = 0
losses = []

optimizer.zero_grad(set_to_none=True)
for i, batch in enumerate(dl):
    ids = batch["input_ids"].to(device)
    labs = batch["labels"].to(device)
    with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype if torch.cuda.is_available() else torch.float32):
        x = model.tgt_embed(ids)
        for layer in model.decoder.layers:
            x, _ = layer(x, tgt_mask=None, use_cache=False)
        x = model.decoder.norm(x)
        logits = model.project(x)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits[...,:-1,:].contiguous().view(-1, VOCAB_SIZE),
            labs[...,1:].contiguous().view(-1)
        ) / GRAD_ACCUM

    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()
    micro += 1
    real_loss = loss.item() * GRAD_ACCUM
    losses.append(real_loss)

    if micro >= GRAD_ACCUM:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        micro = 0
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        print(f"  Step {step}: loss={real_loss:.4f}, lr={lr:.8f}, grad_norm={float(gn):.2f}")

    if step >= 3:
        break

# Check loss decreased
print(f"\n  Loss[0]={losses[0]:.4f}, Loss[-1]={losses[-1]:.4f}")
if losses[-1] < losses[0]:
    print("[PASS] Loss is decreasing!")
else:
    print("[INFO] Loss not yet decreasing (normal for just 3 steps)")

print("\n" + "=" * 50)
print("[PASS] ALL CHECKS PASSED - ready for Azure!")
print("=" * 50)
