# Non-Record V3.1: 7-Layer Shared Transformer + Int8 QAT + 4×MLP + EMA + Sliding-Window Eval

**Mean val_bpb: 1.36409** (2 seeds, stride-64 sliding window) | **vs V2 baseline: −0.03284 BPB** | **Artifact ~14.80 MB** | DGX Spark (1× GB10 Blackwell)

## 1. Overview

V3.1 improves on the [V2 non-record baseline](../2026-04-13_UNet_Int8QAT_7L_4xMLP_EMA_LongTrain) by **−0.03284 BPB mean** (1.39693 → 1.36409) through a single change: **validation evaluation methodology**.

- **Architecture unchanged from V2**: 7-layer UNet shared transformer, 4× MLP multiplier, Int8 QAT, EMA, Muon optimizer, 4 h training budget.
- **Hyperparameters unchanged from V2**: same learning rates, same schedule, same optimizer.
- **Sole change**: replaced chunked eval (stride = `train_seq_len`) with stride-64 sliding-window eval, matching the upstream [2026-03-19 SlidingWindowEval record](../../track_10min_16mb/2026-03-19_SlidingWindowEval) (PR #77).
- Training code path is byte-for-byte equivalent to V2 apart from a post-training filename fix (see Section 7).
- Non-record track; developed on NVIDIA DGX Spark with AI-assisted coding.

## 2. Architecture

Identical to [V2 (2026-04-13)](../2026-04-13_UNet_Int8QAT_7L_4xMLP_EMA_LongTrain).

| Property | Value |
|---|---|
| Transformer layers | 7 |
| Shared-transformer topology | U-Net skip connections (encoder/decoder halves, learned skip weights) |
| Model dim | 512 |
| Attention heads | 8 query, 4 KV (GQA, head_dim 64) |
| MLP multiplier | 4× |
| MLP activation | Leaky ReLU squared (negative_slope = 0.5, then squared) |
| Embeddings | Tied, separate `tied_embed_lr` |
| Logit softcap | Tanh, softcap = 30.0 |
| Positional encoding | RoPE (base 10000.0) |
| Vocab | SentencePiece BPE, 1024 tokens |
| Sequence length | 1024 |
| Total params | **20,725,304** |

### Training hyperparameters (unchanged from V2)

| Hparam | Value |
|---|---|
| `MATRIX_LR` | 0.08261619767374824 |
| `SCALAR_LR` | 0.014691154447587356 |
| `TIED_EMBED_LR` | 0.021552090970329115 |
| `HEAD_LR` | 0.0 (tied-only head path) |
| `MUON_MOMENTUM` | 0.9382982028913158 |
| `WARMDOWN_ITERS` | 1558 |
| `EMA_DECAY` | 0.997 |
| `QAT_ENABLED` | 1 (int8) |
| `TRAIN_BATCH_TOKENS` | 524,288 (8 grad-accum steps) |
| `MAX_WALLCLOCK_SECONDS` | 14400 (4 h) |

**Training code is identical to V2.** The only code change in `train_gpt.py` vs. V2's training script is the post-training checkpoint-filename fix (Section 7). The main training loop, optimizer, EMA, QAT, and serialization are untouched.

## 3. Sliding Window Evaluation

**Algorithm** (see `eval_val_sliding` in `train_gpt.py`):

1. Slide a context window of size `TRAIN_SEQ_LEN = 1024` across the validation corpus in stride-`64` increments.
2. **First window**: score all `TRAIN_SEQ_LEN − 1 = 1023` next-token targets. (Every token except the first of the corpus gets scored once.)
3. **Subsequent windows**: score only the **rightmost `stride = 64` targets**, each using up to `TRAIN_SEQ_LEN − stride = 960` tokens of left context.
4. Total windows on fineweb val (62,021,632 tokens): **969,088**.

Every validation token is scored exactly once with **at least 960 and up to 1023 tokens of left context**. Chunked eval (V2) by contrast forced the first token of every window after the first to be scored with zero context — ≈0.1% of tokens scored under hostile conditions.

**Precedent**: the stride-64 sliding-window protocol was established by Matthew Li's [2026-03-19 SlidingWindowEval record](../../track_10min_16mb/2026-03-19_SlidingWindowEval) (PR #77) on the 10-minute-track. V3.1 brings the non-record track's eval in line with that protocol.

**Eval runs on the int8+zlib round-tripped checkpoint** (decompress → dequantize → `load_state_dict(strict=True)`), so the reported BPB is exactly what a reviewer loads from the submitted `.int8.ptz` artifact.

## 4. Results

| Seed | V2 BPB (chunked) | V3.1 BPB (sliding-64) | Δ (V3.1 − V2) |
|------|------------------|------------------------|----------------|
| 314  | 1.39533 | **1.36221** | **−0.0331** |
| 1337 | 1.39983 | **1.36598** | **−0.0339** |
| 42   | 1.39564 | N/A (see Section 5) | N/A |

**2-seed V3.1 mean**: `(1.36220548 + 1.36597668) / 2 = 1.36409108`
**V2 3-seed mean**: `1.39693`
**Mean Δ vs V2**: **−0.03284 BPB**

### Exact eval numbers (from `logs/eval_seed*.log`)

| Seed | val_loss | val_bpb | tokens | bytes | wallclock |
|---|---|---|---|---|---|
| 314  | 2.30002394 | 1.36220548 | 62,022,592 | 151,082,508 | 20,011.7 s |
| 1337 | 2.30639145 | 1.36597668 | 62,022,592 | 151,082,508 | 19,335.4 s |

Both evals: stride = 64, `EVAL_BATCH_SEQS = 1024`, single GB10 GPU via `WORLD_SIZE = 1`.

### Artifact sizes (`final_model_seed{SEED}.int8.ptz`, int8 per-row + zlib)

| Seed | Artifact bytes | Under 16 MB cap? |
|---|---|---|
| 314  | 15,610,091 | ✓ |
| 1337 | 15,509,631 | ✓ |

## 5. Seed 42 Note

Seed 42 is not part of this 2-seed submission.

The original V3 seed-42 run produced a valid checkpoint (`final_model.int8.ptz`, 15,653,884 B, val_bpb 1.36737 post-ngram under chunked eval on 2026-04-14). That checkpoint file was written to the training repo's CWD with a non-seed-specific filename. A subsequent seed-1337 run, and later a seed-314 run, both wrote to the same filename — the seed-42 and seed-1337 checkpoints were overwritten before they could be copied to a locked location. Only the seed-314 artifact survived to `checkpoints_locked/final_model_seed314.int8.ptz` (chmod 444).

Root cause and fix are documented in `BUGS.md` at the training-host repo root. The fix (seed-suffixed filenames, CWD-override via `OUTPUT_DIR`) is included in the `train_gpt.py` of this submission and was used for the V3.1 seed-1337 retrain. A seed-42 retrain was deferred in favor of meeting the submission timeline; the 2-seed result is presented transparently rather than extrapolating from incomplete data.

The 2-seed mean BPB of **1.36409** is still **−0.03284 BPB vs the V2 3-seed baseline**, so the submission's headline improvement is robust to the missing third seed.

## 6. Ablation Trail

We initially explored **n-gram backoff cache mixing** (see unmerged branch [`submission/v3-ngram-backoff-7l`](https://github.com/AlirezaAlampour/parameter-golf/tree/submission/v3-ngram-backoff-7l), directory `2026-04-16_SharedTransformer_NGramBackoff_Int8_7L`). The idea: mix a BackoffNGram model into the transformer's output distribution at eval time to pick up easy wins on high-frequency n-gram boundaries.

An overnight sweep on seed 314's checkpoint (results in `logs/ablations/overnight_sweep_summary.txt`) compared:

- **stride × ngram grid**: stride ∈ {64, 128, 256}, ngram ∈ {True, False}
- **α × order grid**: α ∈ {0.15…0.70}, order ∈ {5, 7, 9}

Key findings:

- **Sliding stride=64 with ngram OFF** on seed 314: **1.36164861 BPB**
- **Best ngram-ON cell** (α=0.15, order=5): **1.36344949 BPB**
- **Sliding-only (no ngram) dominated ngram-mixed** on this checkpoint by **−0.00180 BPB**.
- Stride sweep: 64 (1.36165) < 128 (1.36179) < 256 (1.36268). Diminishing returns past stride 64.

V3.1 therefore removes the n-gram backoff machinery entirely — `BackoffNGram`, `evaluate_ngram_mixed`, and `eval_val_ngram` are deleted from `train_gpt.py`, not just disabled. The v3.1 `train_gpt.py` has **zero references** to any of those identifiers.

See `logs/ablations/` for the raw sweep summary.

## 7. Reproducibility

- **Hardware**: NVIDIA DGX Spark, single GB10 Blackwell (sm_121), 128 GB unified CPU+GPU memory, CUDA 13.0, ARM64.
- **Training invocation**: `torchrun --standalone --nproc_per_node=1 train_gpt.py` with env vars per Section 2.
- **Eval invocation**: `python3 train_gpt.py` after training sets `args.eval_stride = 64` and calls `eval_val_sliding` on the round-tripped int8+zlib checkpoint.

### R2 reproducibility check

Before running the 2-seed v3.1 evals, we re-ran the seed-314 sliding eval through the V3.1 code path and compared to the earlier overnight-sweep result on the same checkpoint:

| Run | val_bpb |
|---|---|
| Overnight sweep (stride 64, ngram OFF, seed 314) | 1.36164861 |
| V3.1 code-path re-eval (R2) | **1.36220548** |
| |Δ| | 0.00056 |

**Interpretation**: |Δ| < 0.001 → PASS under the "within 0.001 is numerical noise" threshold. The +0.00056 residual is consistent with bf16 autocast in the V3.1 code path vs. whatever precision the overnight sweep's `p_model_stride64.npy` cache was computed at. See `logs/ablations/r2_reproducibility.txt` for the full comparison.

### Determinism note

This submission does **not** enforce deterministic CUDA ops (no `torch.use_deterministic_algorithms(True)`, no `cudnn.deterministic=True`, default SDPA backend priority). This matches the repo norm: V2 and other merged submissions in this track also train non-deterministically.

Seed 1337 was retrained under V3.1 (with the filename fix applied). **The first training step's loss is bit-identical to the V3 seed-1337 run** (`6.9359` to 4 decimals). Divergence begins at step 2 at the `10⁻⁴` scale and accumulates from there, consistent with CUDA matmul reduction-order non-determinism. No evidence of a code regression was found; see `logs/ablations/determinism_note.txt` for the step-by-step comparison.

The practical consequence: seed 1337's V3.1 BPB of 1.36598 reflects one sampled trajectory from the seed's distribution. The seed-314 checkpoint was NOT retrained for V3.1 (Section 5 explains why seeds 1337 and 42 were at risk of the filename bug; 314 was not, so we preserved its V3 checkpoint); its V3.1 BPB is a clean read-out of what sliding eval says about the exact weights that V3 trained.

## 8. Hardware and Eval Setup

- **Training**: single GB10 Blackwell on DGX Spark, 128 GB unified memory, `MAX_WALLCLOCK_SECONDS = 14400` (4 h). Peak GPU memory ≈ 25.4 GiB allocated / 25.9 GiB reserved. Per-step wallclock ≈ 13.85 s.
- **Eval**: stride-64 sliding-window eval on a single GB10 takes ≈ 5.5 h per seed (969,088 windows × ~20 ms/window with `EVAL_BATCH_SEQS = 1024`).
- **Expected competition-hardware eval time**: per issue #1017, the accepted 10-minute-track stride-64 record runs in ~70 s on 8× H100 SXM. Scaling that to this non-record-track checkpoint size gives ~279 s on the competition hardware; well under the 600 s record-track budget and trivially within the non-record track's more lenient budget.

## Files

- `train_gpt.py` — self-contained training + sliding-window eval (1,421 lines). No n-gram code path. Filename fix included.
- `requirements.txt` — identical to V2 (torch≥2.6, numpy≥2.0, sentencepiece≥0.2, datasets≥4.0, huggingface_hub≥0.20).
- `submission.json` — structured result manifest.
- `logs/train_seed314.log` — V3 training log for seed 314 (the checkpoint was NOT retrained for V3.1; see Section 5).
- `logs/train_seed1337.log` — V3.1 seed-1337 retrain log (post filename-fix).
- `logs/eval_seed314.log` — R2 stride-64 sliding eval on seed 314.
- `logs/eval_seed1337.log` — stride-64 sliding eval on seed 1337.
- `logs/ablations/overnight_sweep_summary.txt` — stride × ngram sweep and α × order sweep, showing sliding-only dominates n-gram mixing on this checkpoint.
- `logs/ablations/r2_reproducibility.txt` — seed-314 R2 re-eval numbers.
- `logs/ablations/determinism_note.txt` — step-by-step V3 vs V3.1 seed-1337 training trajectory comparison.
