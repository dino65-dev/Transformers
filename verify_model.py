"""
verify_model.py - sanity check for Flash-Attention + REPO-Attention upgrades.

Run from: f:/anitgravity_code/pre_train_model/Transformers/
    python verify_model.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

OK   = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
all_passed = True


def check(label, condition, detail=""):
    global all_passed
    symbol = OK if condition else FAIL
    print(f"{symbol} {label}  {detail}")
    if not condition:
        all_passed = False
    return condition


# ------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------
print("\n=== Import Check ===")
try:
    from transformer.build_transformer import build_transformer, Transformer
    from transformer.gqa import GroupedQueryAttention
    from transformer.rope import RotaryPositionEmbedding
    from transformer.repo_module import RePoModule
    from transformer.decoder import Decoder
    from transformer.decoder_block import DecoderBlock
    from transformer.projection_layer import ProjectionLayer
    print(f"{OK} All imports succeeded")
except Exception as e:
    print(f"{FAIL} Import failed: {e}")
    sys.exit(1)


# ------------------------------------------------------------
# 2. Flash-Attention availability
# ------------------------------------------------------------
print("\n=== Flash-Attention Check ===")
torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
fa_available = torch_ver >= (2, 0)
check(
    f"PyTorch {torch.__version__}",
    fa_available,
    "Flash-Attention supported" if fa_available else "Fallback to manual attn (upgrade to PyTorch >= 2.0)"
)


# ------------------------------------------------------------
# 3. Build model
# ------------------------------------------------------------
print("\n=== Model Build ===")
VOCAB    = 32000
D_MODEL  = 256    # small for fast CPU verification
N_LAYERS = 2
N_HEADS  = 4
KV_HEADS = 2
D_FF     = 512

try:
    model = build_transformer(
        src_vocab_size=VOCAB,
        tgt_vocab_size=VOCAB,
        src_seq_len=512,
        tgt_seq_len=512,
        d_model=D_MODEL,
        N=N_LAYERS,
        h=N_HEADS,
        kv_h=KV_HEADS,
        d_ff=D_FF,
        dropout=0.0,
        use_repo=True,
        use_flash=fa_available,
    )
    n_params = sum(p.numel() for p in model.parameters())
    check("Model built", True, f"~{n_params/1e6:.2f}M params")
except Exception as e:
    check("Model built", False, str(e))
    import traceback; traceback.print_exc()
    sys.exit(1)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"{INFO} Running on: {device}")


# ------------------------------------------------------------
# 4. Forward pass
# ------------------------------------------------------------
print("\n=== Forward Pass ===")
BATCH, SEQ = 2, 64

try:
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)

    with torch.no_grad():
        embeddings = model.tgt_embed(input_ids)
        check("Embedding shape", embeddings.shape == (BATCH, SEQ, D_MODEL),
              str(tuple(embeddings.shape)))

        x = embeddings
        for li, layer in enumerate(model.decoder.layers):
            x, kv = layer(x, tgt_mask=None, use_cache=False)
            check(f"  Layer {li} output", x.shape == (BATCH, SEQ, D_MODEL),
                  str(tuple(x.shape)))
            check(f"  Layer {li} KV returned", kv is not None,
                  f"kv[0].shape={tuple(kv[0].shape)}")

        x = model.decoder.norm(x)
        check("Final norm", x.shape == (BATCH, SEQ, D_MODEL))

        logits = model.project(x)
        check("Logits shape", logits.shape == (BATCH, SEQ, VOCAB),
              str(tuple(logits.shape)))

        check("No NaN in logits", not torch.isnan(logits).any().item())
        check("No Inf in logits", not torch.isinf(logits).any().item())

except Exception as e:
    check("Forward pass", False, str(e))
    import traceback; traceback.print_exc()
    sys.exit(1)


# ------------------------------------------------------------
# 5. Loss + Backward
# ------------------------------------------------------------
print("\n=== Loss & Backward Pass ===")
model.train()
try:
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
    labels    = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)

    x = model.tgt_embed(input_ids)
    for layer in model.decoder.layers:
        x, _ = layer(x, tgt_mask=None, use_cache=False)
    x = model.decoder.norm(x)
    logits = model.project(x)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.CrossEntropyLoss()(
        shift_logits.view(-1, VOCAB),
        shift_labels.view(-1)
    )

    check("Loss computed", True, f"loss={loss.item():.4f}")
    check("Loss is finite", torch.isfinite(loss).item())

    loss.backward()
    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    check("All gradients finite", grad_ok)

except Exception as e:
    check("Loss & backward", False, str(e))
    import traceback; traceback.print_exc()
    sys.exit(1)


# ------------------------------------------------------------
# 6. REPO-Attention positions
# ------------------------------------------------------------
print("\n=== REPO-Attention Position Check ===")
try:
    model.eval()
    repo_module = model.decoder.layers[0].masked_attention.repo
    check("RePoModule present in layer 0", repo_module is not None)

    x_test = torch.randn(BATCH, SEQ, D_MODEL, device=device)
    with torch.no_grad():
        positions = repo_module(x_test)
    check("RePoModule output shape", positions.shape == (BATCH, SEQ),
          str(tuple(positions.shape)))
    check("Positions are floats", positions.dtype == torch.float32)
    check("Positions vary (not constant)", positions.std().item() > 0,
          f"std={positions.std().item():.4f}")

except Exception as e:
    check("REPO-Attention", False, str(e))


# ------------------------------------------------------------
# 7. Flags check
# ------------------------------------------------------------
print("\n=== Feature Flags ===")
gqa = model.decoder.layers[0].masked_attention
check("use_flash flag", gqa.use_flash == fa_available)
check("use_repo flag",  gqa.use_repo  == True)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\n" + "="*50)
if all_passed:
    print(f"{OK} ALL CHECKS PASSED - model is ready!")
else:
    print(f"{FAIL} SOME CHECKS FAILED - see above")
print(f"   REPO-Attention : {'ON' if gqa.use_repo else 'OFF'}")
print(f"   Flash-Attention: {'ON' if gqa.use_flash else 'OFF (manual fallback)'}")
print("="*50 + "\n")
