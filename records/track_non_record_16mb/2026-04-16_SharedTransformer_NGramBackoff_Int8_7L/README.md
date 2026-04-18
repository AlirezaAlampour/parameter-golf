# Non-Record V3: Shared 7-Layer UNet + Int8 QAT + Eval-Time 5-Order N-Gram Backoff Cache

**V2 stack (7L UNet + MLP×4 + Muon + EMA + Int8 QAT + 4h training) + online 5-order backoff n-gram cache mixed at α=0.2 with the model head at eval time**

**Mean val_bpb: 1.36799** (3 seeds) | **Best seed: 1.36574516** | **Artifact: ~14.99 MB** | DGX Spark (1x GB10 Blackwell, 128GB unified)

> **Non-record submission** — trained on NVIDIA DGX Spark (single GB10 GPU) rather than 8xH100. V3 iteration over the [2026-04-13 v2 submission](../2026-04-13_UNet_Int8QAT_7L_4xMLP_EMA_LongTrain), improving by **0.02894 BPB** (1.39693 → 1.36799, ~2.1%) through a single change: an eval-time 5-order backoff n-gram cache, built online from already-scored validation tokens and mixed with the model head at α=0.2. Architecture, training recipe, and shipped artifact are unchanged from v2. Entire project developed with AI-assisted coding.

## Results

| Metric | Seed 42 | Seed 1337 | Seed 314 |
|--------|---------|-----------|----------|
| val_bpb (int8+zlib roundtrip + n-gram mix, exact) | **1.36736830** | **1.36574516** | **1.37084865** |
| val_loss | 2.30874400 | 2.30600349 | 2.31462052 |
| Steps completed | 1069 | 1061 | 1057 |
| Artifact size (code + compressed model) | 15,715,596 B | 15,692,284 B | 15,671,817 B |
| Training time | 4h wallclock cap | 4h wallclock cap | 4h wallclock cap |

**Cross-seed statistics**: mean 1.36799 BPB, range 0.00510 BPB (0.37% of mean), std dev ~0.00258 BPB. Spread matches v2's noise level, confirming the gain comes from the n-gram addition rather than seed luck.

## Improvement over V2

| Dimension | V2 (2026-04-13) | V3 (2026-04-16) |
|-----------|-----------------|-----------------|
| Architecture | 7L UNet, d=512, MLP×4 | **unchanged** |
| Training wallclock | 4 h | 4 h |
| Steps reached | ~1037 (mean) | ~1062 (mean) |
| Optimizer / QAT / EMA | Muon + Int8 QAT + EMA(0.997) | **unchanged** |
| Quantization / compression | Int8 per-row + zlib | **unchanged** |
| Eval path | model head only | **model head + 5-order backoff n-gram, α=0.2** |
| Shipped artifact (code + model) | ~15.55 MB | ~14.99 MB (mean) — essentially unchanged |
| Best val_bpb | 1.39533 | **1.36575** |
| Mean val_bpb | 1.39693 | **1.36799** |

The full 0.02894 BPB gain comes from the eval-time n-gram cache. No extra parameters or training compute were added.

## Eval-Time 5-Order N-Gram Backoff Cache

Implemented in [`train_gpt.py`](train_gpt.py) as `class BackoffNGram` and consumed by `evaluate_ngram_mixed`. The cache is built fresh at eval time from scored validation tokens and is **not** serialized into the 16 MB artifact.

### Algorithm

At each validation position `i`:

1. Compute `p_model = softmax(model_logits)[target_i]` under `torch.inference_mode()`.
2. Look up `p_ngram` from the backoff cache, conditioned on the up-to-4-token prefix ending at `i-1`. The cache falls back from order-5 → 4 → 3 → 2 → 1 → unigram (with Laplace smoothing, λ=0.01) and returns the first order with a non-empty counter.
3. If any order ≥ 2 hit, mix: `p_final = 0.8 · p_model + 0.2 · p_ngram`. Otherwise, `p_final = p_model`.
4. Accumulate `-log p_final` into the token-level NLL.
5. **Only then** call `ngram.observe(prefix, target_i)` to record the target into the cache for future positions.

This is the "legal score-first" pattern (cf. [2026-04-06_SP8192_QK5_LegalTTT](../../track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828)): scoring position `i` uses only the cache built from positions `< i`. No future-token leakage.

### Hyperparameters

- `max_order = 5`
- `mix_ngram = 0.2`, `mix_model = 0.8`
- `laplace = 0.01`
- `block_size = 1024`, `stride = 512` (same sliding-window scheme as the pre-quant eval)

### Hit-rate distribution (seed 42, 62,021,632 scored tokens)

| Matched order | Count | % |
|---:|---:|---:|
| 0 (no hit) | 1 | 0.0% |
| 1 (unigram) | 885 | 0.001% |
| 2 | 292,873 | 0.47% |
| 3 | 5,619,669 | 9.06% |
| 4 | 15,139,277 | 24.41% |
| **5 (longest)** | **40,968,927** | **66.06%** |

Two-thirds of tokens found an exact 4-token prefix match in the online cache, with the 5-gram probability dominating the mix. The wall-clock cost of the n-gram pass is ~40 minutes per seed on GB10.

### Artifact impact

The cache adds **zero bytes** to the shipped artifact. It is a Python `defaultdict[tuple[int, ...], Counter]` constructed inside `evaluate_ngram_mixed` and released when the function returns. Only the int8+zlib model file ships:

| Seed | model int8+zlib | code (train_gpt.py) | **total** | headroom vs 16 MB |
|------|---:|---:|---:|---:|
| 42 | 15,653,884 | 61,712 | 15,715,596 | 1,061,620 B |
| 1337 | 15,630,558 | 61,726 | 15,692,284 | 1,084,932 B |
| 314 | 15,610,091 | 61,726 | 15,671,817 | 1,105,399 B |

## Architecture (unchanged from v2)

- **7 transformer layers**, dim=512, 8 heads, 4 KV heads (GQA), head_dim=64
- **MLP multiplier 4×**
- **U-Net skip connections**: encoder/decoder halves with learned skip weights and per-block residual mixing from input embedding
- **Leaky ReLU squared** MLP activation (negative_slope=0.5, then squared)
- **Tied embeddings** with separate tied_embed_lr
- **Logit softcap** (tanh-based)
- **RoPE** positional encoding
- Vocab size 1024 (SentencePiece BPE)
- Sequence length 1024
- 20,725,304 parameters

## Training Hyperparameters (unchanged from v2)

- MATRIX_LR: 0.08261619767374824
- SCALAR_LR: 0.014691154447587356
- TIED_EMBED_LR: 0.021552090970329115
- HEAD_LR: 0.0 (tied-only head path)
- MUON_MOMENTUM: 0.9382982028913158
- WARMDOWN_ITERS: 1558
- EMA_DECAY: 0.997 (EMA enabled)
- QAT_ENABLED: 1 (int8)
- 524,288 train tokens per step, 8 gradient accumulation steps
- MAX_WALLCLOCK_SECONDS: 14400 (4 hours)

## Reproduction

```bash
# From the submission directory, with fineweb10B_sp1024 preprocessed at
# ./data/datasets/fineweb10B_sp1024/ and the tokenizer at
# ./data/tokenizers/fineweb_1024_bpe.model:

SEED=42 \
NUM_LAYERS=7 MLP_MULT=4 \
MATRIX_LR=0.08261619767374824 \
SCALAR_LR=0.014691154447587356 \
TIED_EMBED_LR=0.021552090970329115 \
HEAD_LR=0.0 \
MUON_MOMENTUM=0.9382982028913158 \
WARMDOWN_ITERS=1558 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
QAT_ENABLED=1 USE_INT6=0 \
MAX_WALLCLOCK_SECONDS=14400 \
python3 train_gpt.py
```

Repeat with `SEED=1337` and `SEED=314` for the other two seeds. Total wallclock per seed: ~4h 40min (4h training + ~40min n-gram eval).

## Quantization & Serialization (unchanged from v2)

- **Int8 per-row quantization** for all weight matrices
- **Float16 passthrough** for small tensors and control parameters
- **zlib compression** on serialized checkpoint
- Roundtrip validation: decompress, dequantize, and re-evaluate under the n-gram-mixed metric
- Mean artifact ~15.67 MB — well under the 16 MB cap

## Hardware

- **NVIDIA DGX Spark**: ARM64 (aarch64), GB10 GPU (Blackwell sm_121)
- 128 GB unified CPU+GPU memory (no discrete VRAM)
- CUDA 13.0
- Single GPU training via `python3 train_gpt.py`
- Training: ~13.5 s/step, ~1062 steps in 4 h wallclock (near-identical across all 3 seeds)
- N-gram eval: ~40 minutes per seed on the full `fineweb_val_*` split (62,021,632 tokens)
- Peak GPU memory: ~25.5 GB allocated / ~27.5 GB reserved

## Seed 42 note

Seed 42 was trained on 2026-04-13 with an earlier revision of the script that had a `tgt_ids` dtype bug in the byte-accounting step of `eval_val_ngram`. Training and int8+zlib serialization completed successfully (15,653,884 B artifact), but the post-training eval crashed before printing the final BPB. After the one-line dtype fix (cast to `.long()` before indexing byte LUTs), the exact same serialized artifact was re-evaluated and produced the number reported above (1.36736830). The recovery's n-gram hit-rate histogram matches the original run's exactly, confirming deterministic recovery. Seeds 1337 and 314 were trained with the fixed code and completed normally. Full recovery trace is appended to [`train_seed42.log`](train_seed42.log).

## Development

This submission was developed with AI-assisted development using Claude for architecture exploration, hyperparameter tuning, and orchestration of the multi-seed training runs. All training and evaluation ran on DGX Spark hardware.

## Files

- `train_gpt.py` — self-contained training + eval script (same config across all 3 seeds)
- `train_seed42.log` / `train_seed1337.log` / `train_seed314.log` — raw training logs
- `v3_3seed_summary.txt` — collector output summarizing the three seeds
- `submission.json` — structured result manifest
- `requirements.txt` — pip dependencies
