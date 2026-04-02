import math
total_steps = 6_000_000_000 // (24 * 8 * 2048)
total_updates = total_steps * 1
warmup = 2000
min_lr_ratio = 0.1
peak_lr = 3e-4

print(f"Estimated steps per epoch: {total_steps:,}")
print(f"Warmup steps: {warmup}")
print()
header = f"{'Step':>8} {'Scale':>8} {'LR':>12}  Phase"
print(header)
print("-" * 48)
for step in [0, 100, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 12000, 15000]:
    s = step + 1
    if s <= warmup:
        scale = float(s) / float(max(1, warmup))
        phase = "warmup"
    else:
        progress = (s - warmup) / max(1, total_updates - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        phase = "cosine"
    lr = peak_lr * scale
    print(f"{step:>8} {scale:>8.4f} {lr:>12.8f}  {phase}")
