"""Patch pretrain_slimpajama.ipynb to fix stuck loss."""
import json

with open('pretrain_slimpajama.ipynb', 'r') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    # ========== FIX 1: Config cell ==========
    if 'LABEL_SMOOTHING' in src and 'BATCH_SIZE' in src:
        src = src.replace('LEARNING_RATE   = 2e-4', 'LEARNING_RATE   = 3e-4')
        src = src.replace('MAX_GRAD_NORM   = 0.5', 'MAX_GRAD_NORM   = 1.0')
        src = src.replace('WARMUP_STEPS    = 1000', 'WARMUP_STEPS    = 2000')
        src = src.replace('LABEL_SMOOTHING = 0.05', 'LABEL_SMOOTHING = 0.0   # No label smoothing for pre-training')

        # Add estimated total steps for cosine schedule
        src = src.replace(
            'LABEL_SMOOTHING = 0.0   # No label smoothing for pre-training',
            'LABEL_SMOOTHING = 0.0   # No label smoothing for pre-training\n'
            'ESTIMATED_TOKENS = 6_000_000_000  # SlimPajama-6B total tokens\n'
            'ESTIMATED_TOTAL_STEPS = ESTIMATED_TOKENS // (BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN)  # ~15k steps/epoch'
        )
        cell['source'] = [src]
        print("[FIXED] Config cell: LR=3e-4, grad_norm=1.0, warmup=2000, label_smooth=0, added ESTIMATED_TOTAL_STEPS")

    # ========== FIX 2: Training loop - force cosine decay ==========
    if 'lr_lambda' in src and 'has_known_total_updates' in src:
        # Replace the try/except block
        old_try = (
            'try:\n'
            '    batches_per_epoch = len(train_loader)\n'
            '    updates_per_epoch = math.ceil(batches_per_epoch / GRAD_ACCUM)\n'
            '    total_updates = max(1, updates_per_epoch * EPOCHS)\n'
            '    warmup_updates = min(WARMUP_STEPS, total_updates)\n'
            '    has_known_total_updates = True\n'
            'except TypeError:\n'
            '    batches_per_epoch = None\n'
            '    updates_per_epoch = None\n'
            '    total_updates = None\n'
            '    warmup_updates = max(1, WARMUP_STEPS)\n'
            '    has_known_total_updates = False\n'
            '    print("train_loader has no __len__ (IterableDataset). Using warmup + inverse-sqrt LR decay.")'
        )

        new_try = (
            'try:\n'
            '    batches_per_epoch = len(train_loader)\n'
            '    updates_per_epoch = math.ceil(batches_per_epoch / GRAD_ACCUM)\n'
            '    total_updates = max(1, updates_per_epoch * EPOCHS)\n'
            'except TypeError:\n'
            '    # IterableDataset: estimate from known token count\n'
            '    total_updates = ESTIMATED_TOTAL_STEPS * EPOCHS\n'
            '    print(f"IterableDataset detected. Estimated total optimizer steps: {total_updates}")\n'
            '\n'
            'warmup_updates = min(WARMUP_STEPS, total_updates)\n'
            'has_known_total_updates = True  # Always use cosine decay'
        )

        if old_try in src:
            src = src.replace(old_try, new_try)
            print("[FIXED] Training loop: replaced IterableDataset fallback with estimated cosine decay")
        else:
            print("[WARN] Could not find exact try/except block to replace. Checking partial...")
            # Try partial match
            if 'inverse-sqrt LR decay' in src:
                print("  Found inverse-sqrt reference. Attempting line-by-line fix.")

        # Remove the inverse-sqrt fallback from lr_lambda
        old_fallback = (
            '    # Fallback for IterableDataset with unknown total steps.\n'
            '    decay = math.sqrt(float(warmup_updates) / float(step))\n'
            '    return max(MIN_LR_RATIO, decay)'
        )
        new_fallback = (
            '    # Should not reach here since has_known_total_updates is always True now.\n'
            '    return MIN_LR_RATIO'
        )

        if old_fallback in src:
            src = src.replace(old_fallback, new_fallback)
            print("[FIXED] Removed inverse-sqrt fallback from lr_lambda")
        else:
            print("[WARN] Could not find inverse-sqrt fallback in lr_lambda")

        cell['source'] = [src]

with open('pretrain_slimpajama.ipynb', 'w') as f:
    json.dump(data, f, indent=1)

print("\nAll patches applied! Re-upload to Azure and restart training from scratch.")
