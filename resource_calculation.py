#!/usr/bin/env python3
"""
Resource counting for a GPT-2 XL-style decoder Transformer.

  (a) Parameter count and fp32 memory footprint
  (b) Forward-pass FLOPs for a single sequence
  (c) Which component of a block dominates the FLOPs

Edit the CONFIG block and re-run to get the numbers for any model.

"""




# ------------------------------- CONFIG -------------------------------
vocab_size     = 50_257
context_length = 16384
num_layers     = 48
d_model        = 1_600
num_heads      = 25
d_k            = 64       # per-head dim; ideally num_heads * d_k == d_model
d_ff           = 4_288    # nearest multiple of 64 to (8/3)*d_model = 4266.7
# ----------------------------------------------------------------------

seq             = context_length   # sequence length for one forward pass
d_v             = d_k              # value dim per head (same as key dim here)
BYTES_PER_PARAM = 4               # fp32 weights -> 4 bytes each

# --------Small --------
# num_layers     = 12
# d_model =  768
# num_heads = 12

# --------Medium --------
# num_layers     = 24
# d_model =  1024
# num_heads = 16

# --------Large --------
# num_layers     = 36
# d_model =  1280
# num_heads = 20

# --------XLarge --------
num_layers     = 48
d_model =  1600
num_heads = 25

def compute_d_ff(d_model, multiple=64):
    return round(8 * d_model / 3 / multiple) * multiple
d_ff = compute_d_ff(d_model)

def mm(m, n, p):
    """FLOPs of an (m x n) @ (n x p) matmul = 2*m*n*p."""
    return 2 * m * n * p


def fmt(n):
    return f"{n:,}"


def human(n, unit=""):
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(n) >= div:
            return f"{n / div:.2f} {suf}{unit}"
    return f"{n:.0f}{unit}"


# ============================ PART (a) ============================
def part_a():
    print("=" * 64)
    print("(a) PARAMETERS & MEMORY")
    print("=" * 64)

    embedding = vocab_size * d_model

    rmsnorm   = 2 * d_model            # two RMSNorms per block
    attn      = 4 * d_model * d_model  # W_Q, W_K, W_V, W_O
    swiglu    = 3 * d_model * d_ff     # W1, W2, W3
    per_block = rmsnorm + attn + swiglu
    blocks    = num_layers * per_block

    final_norm = d_model
    lm_head    = vocab_size * d_model

    total  = embedding + blocks + final_norm + lm_head
    memory = total * BYTES_PER_PARAM

    print(f"  Token embedding  (V*d)        = {fmt(embedding)}")
    print(f"  Per block:")
    print(f"    RMSNorm   (2*d_model)       = {fmt(rmsnorm)}")
    print(f"    Attn QKVO (4*d_model^2)     = {fmt(attn)}")
    print(f"    SwiGLU    (3*d_model*d_ff)  = {fmt(swiglu)}")
    print(f"    -> per block                = {fmt(per_block)}")
    print(f"  All {num_layers} blocks                 = {fmt(blocks)}")
    print(f"  Final RMSNorm  (d_model)      = {fmt(final_norm)}")
    print(f"  LM head        (V*d)          = {fmt(lm_head)}")
    print("  " + "-" * 44)
    print(f"  TOTAL PARAMETERS              = {fmt(total)}  ({human(total)})")
    print(f"  MEMORY (fp32)                 = {fmt(memory)} bytes  ({memory/1e9:.2f} GB)")
    print()
    return total


# ============================ PART (b) ============================
def part_b():
    print("=" * 64)
    print(f"(b) FORWARD-PASS FLOPs   (seq = {seq})")
    print("=" * 64)

    qkv    = 3 * mm(seq, d_model, d_model)   # Q, K, V projections
    scores = num_heads * mm(seq, d_k, seq)    # Q @ K^T  per head
    attnv  = num_heads * mm(seq, seq, d_v)    # softmax @ V  per head
    out    = mm(seq, d_model, d_model)        # output projection
    ffn    = 3 * mm(seq, d_model, d_ff)       # SwiGLU W1, W2, W3

    per_block = qkv + scores + attnv + out + ffn
    blocks    = num_layers * per_block
    lm_head   = mm(seq, d_model, vocab_size)
    total     = blocks + lm_head

    print(f"  Per block:")
    print(f"    Q,K,V projection       = {fmt(qkv):>18}   ({human(qkv)})")
    print(f"    Attention scores QK^T  = {fmt(scores):>18}   ({human(scores)})")
    print(f"    Attention (scores x V) = {fmt(attnv):>18}   ({human(attnv)})")
    print(f"    Output projection      = {fmt(out):>18}   ({human(out)})")
    print(f"    SwiGLU FFN             = {fmt(ffn):>18}   ({human(ffn)})")
    print(f"    -> per block           = {fmt(per_block):>18}   ({human(per_block)})")
    print(f"  All {num_layers} blocks            = {fmt(blocks):>18}   ({human(blocks)})")
    print(f"  LM head                  = {fmt(lm_head):>18}   ({human(lm_head)})")
    print("  " + "-" * 44)
    print(f"  TOTAL FLOPs              = {fmt(total):>18}  ({human(total)})")
    print()
    return per_block, {"qkv": qkv, "scores": scores, "attnv": attnv, "out": out, "ffn": ffn}


# ============================ PART (c) ============================
def part_c(per_block, parts):
    print("=" * 64)
    print("(c) WHAT DOMINATES A BLOCK")
    print("=" * 64)

    rows = [
        ("Q,K,V projection", parts["qkv"]),
        ("Attention scores", parts["scores"]),
        ("Attention x V",    parts["attnv"]),
        ("Output proj",      parts["out"]),
        ("SwiGLU FFN",       parts["ffn"]),
    ]
    biggest = max(rows, key=lambda r: r[1])[0]
    for name, val in rows:
        mark = "  <-- biggest" if name == biggest else ""
        print(f"  {name:<18}{human(val):>9}  {val/per_block*100:6.2f}%{mark}")

    proj  = parts["qkv"] + parts["out"]
    score = parts["scores"] + parts["attnv"]
    ffn   = parts["ffn"]
    print()
    print("  Grouped:")
    print(f"    SwiGLU FFN             {ffn  / per_block * 100:6.2f}%")
    print(f"    Attention projections  {proj / per_block * 100:6.2f}%")
    print(f"    Attention scores       {score/ per_block * 100:6.2f}%")
    print()
    print(f"  => SwiGLU FFN is {ffn/per_block*100:.2f}% of every block, so it dominates.")
    print()


def main():
    if num_heads * d_k != d_model:
        print(f"[warn] num_heads*d_k ({num_heads*d_k}) != d_model ({d_model})\n")
    part_a()
    per_block, parts = part_b()
    part_c(per_block, parts)


if __name__ == "__main__":
    main()