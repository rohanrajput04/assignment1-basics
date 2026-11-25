# Blog plan: CS336 Assignment 1 → a 4-part series

> **How to use this file:** point Claude at it from the blog repo
> (`rohanrajput04.github.io`) as the implementation brief. Every factual claim below is
> sourced to a line in this repo, at
> `/Users/rohanrajput/rohan_apps/cs336/assignments/assignment1-basics`. Read those sources
> rather than trusting the summaries here — and see **Fact-check before publishing**, which
> lists known errors in the source material that must not be copied forward.

## Context

Rohan finished CS336 Assignment 1 (BPE tokenizer, Transformer LM, training loop, ablations,
leaderboard run) and wants to blog about it at https://rohanrajput04.github.io.
**Rohan writes the prose — this file supplies structure, material, and sourcing.**

The site is Quarto-based and already runs one 3-part series ("Parallelism in AI", Feb–Mar 2026:
data parallelism → pipeline → tensor). House style, from the Part 1 post: ~800–900 words, `we`
voice, on-page TOC, heavy on visuals (6 Manim GIFs), no code blocks, no LaTeX, closes with a
References section linking papers.

The assignment writeup (`writeup.tex`, 669 lines) has far more material than one post, and it's
full of measured numbers — so: **4 posts, results-driven, using the 8 W&B PNGs already in
`figures/` plus tables.** This drops the "no tables/no numbers" part of the house style, which is
fine — the parallelism Part 1 post already uses a comparison table.

Source repo: https://github.com/rohanrajput04/assignment1-basics (all on `main`).
W&B experiment log: https://wandb.ai/rohan0401/cs336-tinystories/reports/Experiment-Log--VmlldzoxNzUyMDYxMw

---

## Series structure

Working title: **"Building a Language Model from Scratch"** (Parts 1–4). Every post ends with a
one-line teaser for the next, matching the parallelism series.

### Part 1 — The tokenizer is a compression algorithm

*Hook:* at the throughput I measured, tokenizing the Pile (825 GB) would take ~9 days. That number
makes tokenization feel like real engineering rather than a preprocessing detail.

Sections:
1. **Why bytes, not characters** — UTF-8 is variable-width, self-synchronizing, endianness-free
   (`writeup.tex:80`).
2. **Training BPE** — parallel pre-tokenization + the merge loop. TinyStories 10k vocab: 43.7 s,
   0.18 GB peak. Profile split: pre-tokenize 19 s wall / ~200 s CPU across 12 workers, merge loop
   25 s wall (`writeup.tex:111-121`). The point: it's embarrassingly parallel up front, serial after.
3. **The vocabulary is a mirror of your corpus** — strongest idea in this post. TinyStories' longest
   token is `b' accomplishment'`, a real word; OpenWebText's is a **64-byte mojibake whitespace
   scraping artifact** (`writeup.tex:129-137`). Data quality shows up directly in the vocab.
4. **Compression numbers** — TinyStories 10k: 4.05 bytes/token; OWT 32k: 4.25. A 3× larger vocab
   buys almost nothing, because OWT spends it on variety rather than compression
   (`writeup.tex:155`). Cross-domain: OWT text through the TinyStories tokenizer drops to 3.16 —
   text fragments back toward byte-level tokens (`writeup.tex:161-163`).
5. **Storage** — `uint16` because the whole vocab fits in 0–65535 (`writeup.tex:174`).

Visuals: one small table (vocab / bytes-per-token / longest token). No existing PNG covers this; a
simple bar chart is optional, not required.

Code to reference if useful: `cs336_basics/train_bpe.py`, `cs336_basics/tokenizer.py`,
`cs336_basics/pretokenization_example.py`.

### Part 2 — Where the FLOPs actually go

*Hook:* a single H100 at 50% MFU would need **4,854 hours — about 202 days — to train GPT-2 XL**
(`writeup.tex:424-426`). This is the natural bridge to the existing parallelism series; link it
explicitly at the close.

Sections:
1. **The components** — RMSNorm, SwiGLU, RoPE, causal MHA, pre-norm block (one paragraph each, no code).
2. **Parameter count** — GPT-2 XL: 1.64 B params, 6.56 GB just to load in fp32 (`writeup.tex:265-269`).
3. **FLOP breakdown per block** — the table at `writeup.tex:282-293`. FFN ≈ 60%, QKVO projections
   ≈ 30%, the actual attention matmuls ≈ 10%. Counterintuitive enough to be the post's spine:
   *attention is not where the compute goes.*
4. **How that shifts with model size** — table at `writeup.tex:314-317`: attention's share falls
   18.2% (small) → 9.6% (XL), because projections scale as d² while attention scales as seq²·d.
5. **How it shifts with context** — the punchline. seq 1024 → 16,384 is a 16× increase but a **38×**
   FLOP increase (3.52 → 133.6 TFLOPs), and attention's share goes **10% → 63%** (`writeup.tex:328`).
   Exactly reverses the previous section. This is why long context is its own engineering problem.
6. **Memory** — 16× params for weights+grads+Adam state, so GPT-2 XL is
   `16.37 · batch + 26.24 GB` → **max batch size 3 on an 80 GB card** (`writeup.tex:396-404`).

Visuals: two tables (per-block FLOPs; share by model size). One new chart worth making — attention
share vs. context length — is the highest-value new asset in the whole series.

Code to reference: `cs336_basics/model.py`, `resource_calculation.py`, `resource_counting.md`.

### Part 3 — Training dynamics: what actually moved the loss

*Hook:* four architecture choices, ranked by how much each one is worth in validation loss. Baseline
is 1.564 on TinyStories throughout.

Sections:
1. **Edge of stability** — LR sweep is U-shaped: 1e-4 → 2.32, 3e-4 → 1.88, **1e-3 → 1.56**,
   3e-3 → 1.86, 1e-2 → 2.55 (`writeup.tex:503`). The best LR sits just below where training breaks;
   the fastest useful rate is the largest one that doesn't destabilize. → `figures/lr_sweep.png`
2. **Batch size** — 1 → ~2.6–3.0, 16 → 1.91, 64 → 1.58, **128 → 1.48**, 256 → diverged, with
   `lr = 1e-3·√(B/32)` (`writeup.tex:529-540`). Two honest caveats to keep: this is fixed-*steps*,
   not fixed-*compute*, so bigger batches also just did more work; and the √-scaling overshot at 256.
   → `figures/batch_size.png`
3. **The ablation ranking** — the centerpiece. Present as one table, ordered by damage:

   | removed | val loss | Δ vs baseline |
   |---|---|---|
   | RoPE → NoPE | 1.802 | +0.24 |
   | pre-norm → post-norm | 1.769 | +0.21 |
   | RMSNorm → none | 1.715 | +0.15 |
   | SwiGLU → SiLU | 1.642 | +0.08 |

   (`writeup.tex:580, 593, 606, 620`) → the four `figures/ablation_*.png`
4. **Two results worth dwelling on** — (a) the no-norm model *still trained stably* at the baseline
   LR at this scale, so RMSNorm bought quality, not stability (`writeup.tex:580`); (b) NoPE degrades
   but doesn't collapse, because causal masking leaks a weak positional signal through
   different-sized prefixes (`writeup.tex:606`). Both are more interesting than the headline numbers.
5. **Sample output** — the fairy story at `writeup.tex:556-565`, plus the temperature/top-p tradeoff
   (τ=0.1 repetitive, τ=1.0 incoherent, τ=0.8 + top-p 0.9 the sweet spot).

Visuals: 6 existing PNGs + the ablation table. This post is the best-supported of the four.

Code to reference: `cs336_basics/train.py`, `cs336_basics/generate.py`.

### Part 4 — 45 minutes on a B200

*Hook:* a fixed wall-clock budget turns model training into a throughput problem, and the thing that
mattered most wasn't the architecture.

Sections:
1. **The rules** — B200, ≤45 min, OWT only, validated at context length 512, 32k vocab, ~3.3 to beat.
2. **The naive attempt** — 4 layers / d_model 512 reached only ~4.5, and was validated at the wrong
   context length (`writeup.tex:658`).
3. **Scaling up** — GPT-2-small-shaped: 12 layers, d_model 768, 12 heads, d_ff 2048, ~134 M params,
   SwiGLU + RoPE + RMSNorm, context 512; AdamW + cosine (1000 warmup), grad clip 1.0, lr_max 8e-4,
   batch 64.
4. **Throughput work** — `torch.compile`, bf16 autocast, TF32 matmuls, memory-mapped `uint16`
   streaming. **The actual lever: batch 32 → 64 gave ~2.2× tokens/sec**, because the run was
   overhead/data-loading bound rather than compute bound — that's what fit ~0.7 B tokens into the
   budget (`writeup.tex:658`).
5. **Result, stated honestly** — 3.42 at the 45-minute mark, best single-batch estimate 3.34,
   fluctuating 3.34–3.47. That's *at* the ~3.3 baseline, not in the competitive ~3.0 range. Say so.
6. **Why the TinyStories number isn't comparable** — ~1.5 vs ~4.58 at equal compute is not a fair
   comparison: different vocab means a different entropy floor (ln 32000 ≈ 10.4 vs ln 10000 ≈ 9.2),
   and web text is intrinsically less predictable (`writeup.tex:636-638`).
7. **Infrastructure** — Modal for cloud GPUs, W&B for logging, sweeps fanned out via `train.spawn`
   (`modal_train.py:115-136`). Worth a short section; it's reusable advice.

Visuals: `figures/owt_curve.png`, `figures/leaderboard_curve.png`.

---

## Assets already available

All 8 PNGs live in `figures/` (30–80 KB each, web-ready): `lr_sweep`, `batch_size`,
`ablation_nope`, `ablation_prenorm`, `ablation_rmsnorm`, `ablation_swiglu`, `owt_curve`,
`leaderboard_curve`. Copy them into each post's Quarto folder rather than hotlinking across repos.

---

## Fact-check before publishing

The writeup has a few loose ends. Fix these in the posts rather than copying them forward:

1. **OWT small-model loss is stated twice, inconsistently** — 4.58 at `writeup.tex:636` vs 4.48 at
   `writeup.tex:658`. Pick one from W&B.
2. **The Pile estimate doesn't recompute** — 825 GB ÷ 1.04 MB/s ≈ 9.2 days, but `writeup.tex:168`
   says 9.9. Since it's Part 1's hook, redo the arithmetic and state the assumption (1000 vs 1024 MB/GB).
3. **The leaderboard 3.42 is a single-batch estimate**, not a full validation pass. Label it as such.
4. **Batch size 1 is a range** (~2.6–3.0), not a point estimate — present it as approximate.
5. **Don't cite the Unicode answers** (`writeup.tex:71-93`) — several are placeholder-quality and one
   is empty. No loss; they aren't blog material.

## One judgment call

The repo is a public fork of the Stanford course assignment and now contains full solutions. Writing
about *approach and results* is uncontroversial; prominently linking a public solutions repo may sit
badly with course policy on posting solutions. Worth a deliberate decision before hitting publish —
the posts stand on their own without the repo link.

## Verification

- Draft each post as `posts/<slug>/index.qmd` with frontmatter matching the existing series
  (`title`, `author: Rohan Singh Rajput`, `date`, `categories`, `toc: true`).
- `quarto preview` locally: confirm figures resolve, tables render, TOC populates, and the
  next-post teaser links work once each part is live.
- Cross-check every number against `writeup.pdf` and the W&B report before publishing — the five
  items above are the known discrepancies, and both loss tables should be re-read against W&B.
- Word-count check: target 800–1200 per post to stay near house style. Part 3 is the one at risk of
  running long; if it does, split the ablations off from the LR/batch-size material.
