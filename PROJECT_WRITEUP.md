# Parameter Golf: Competing in OpenAI's ML Challenge with Zero ML Background

> A portfolio writeup of my entry into OpenAI's Parameter Golf competition —
> a byte-level language modeling benchmark scored in bits-per-byte (BPB)
> on a held-out FineWeb validation slice. Lower is better.
>
> **Result:** 1.6656 → 1.310 BPB across four submission iterations,
> running entirely on a single NVIDIA DGX Spark workstation
> (GB10, 128 GB UMA, sm_121 Blackwell), not the 8×H100 reference rig
> the leaderboard targets.
>
> **Background going in:** zero. I had not trained a neural network before
> this project.

📺 **Video walkthrough:** *(YouTube link to be added once published)*

📸 **Dashboard screenshots:** see [`screenshots/`](screenshots/) — progression,
sweep heatmaps, run audit, and the leaderboard gap.

---

## 1. The journey, in one paragraph

I started the competition without knowing what a transformer block was.
I finished four iterations later with a 1.310 BPB submission —
a 21% relative improvement over my first non-record submission —
built on int6 quantization-aware training, sliding-window evaluation,
EMA weight averaging, and the Muon optimizer.
The route there was a multi-LLM workflow:
**Claude** carried the strategy and project memory,
**GPT** wrote and refactored the actual training code,
**Gemini** scraped competitive intel from the public leaderboard and PRs,
**Perplexity** dug up the citations behind techniques I'd never heard of.
I was the integrator, the operator, and the one who kept the dashboard honest.

| Submission | val_bpb | Headline change | Date |
|------------|---------|-----------------|------|
| v1 | 1.6656 | First non-record submission — baseline GPT, no tricks | Apr 7 |
| v2 | 1.3969 | 7L UNet + Int8 QAT + EMA, 4h training, 3-seed mean | Apr 13 |
| v3 | 1.362  | Sliding-window eval (stride=64), n-gram backoff explored & discarded | Apr 18 |
| v4 | 1.310  | Int6 QAT (LZMA) + 10h training + sliding window — single seed | Apr 26 |

---

## 2. The multi-LLM workflow (for non-ML readers)

I used four AI tools, each playing a different position. The point of the
section below isn't to brag about tool choice — it's that **the gains came
from the orchestration, not from any single model.**

| Tool | Role | What it was actually good at |
|------|------|------------------------------|
| **Claude** | Strategist & memory | Long-horizon planning across days, remembering what we tried, structuring ablations, talking me out of bad ideas |
| **GPT** | Code mechanic | Concrete code edits, quick refactors, debugging tracebacks, writing training loops |
| **Gemini** | Competitive intel | Reading the public leaderboard PRs, summarizing what record-holders did differently |
| **Perplexity** | Citation engine | "Where does this technique come from?" — papers, blog posts, original sources |

### What I learned (non-technical)

1. **AI models are better as a team than solo.** Asking one model to do
   strategy + code + research is how you get confident-sounding wrong answers.
   Splitting the work makes their disagreements visible, which is where
   most of the real signal lives.
2. **Each model has different strengths**, and those strengths drift over
   time as new versions ship. The workflow has to be re-checked, not
   memorized.
3. **The scientific method matters more than the model.** Every change
   is a hypothesis. Run it, log it, compare to the previous best, and
   *throw it out if it doesn't beat the baseline.* Most of my "good ideas"
   died this way. The few that survived are the submissions above.
4. **Operator discipline beats raw compute.** I was on one Spark; the
   leaderboard is on 8×H100 racks. I couldn't out-throughput anyone, so
   the only edge available was running cleaner experiments and not
   fooling myself about what worked.

### The checkpoint discovery story

Halfway through the v3 cycle, I spent a full day chasing a sweep over
n-gram backoff parameters (α and max_order) on top of sliding-window
evaluation. The grid is in [`screenshots/sweep_heatmap.png`](screenshots/).
Forty-two cells, seven α values × three max_orders × two stride settings.

The best n-gram cell came in at **val_bpb 1.36345**. The best
sliding-only cell — the same checkpoint, no n-gram backoff at all —
was **1.36165**. The "fancy" technique was *worse than doing nothing on top
of the simple one.* I had been about to submit a v3 with n-gram backoff
turned on. I didn't, because the heatmap said don't.

That single negative result is the cleanest example I have of why the
multi-LLM + dashboard discipline matters. Claude pushed me to actually
run the ablation instead of trusting the paper. The dashboard surfaced
the answer in a way I couldn't argue with.

---

## 3. Technical summary (for ML readers)

### Architecture

- **Backbone:** 7-layer transformer with U-Net-style skip connections
  between mirrored blocks (3↔5, 2↔6, 1↔7), residual gating learned per layer.
- **Width / heads:** tuned per submission; v4 is the int6-friendly variant
  trained from scratch under QAT rather than post-hoc quantized.
- **Activation:** LeakyReLU² in the MLPs (squared, with the leak preserved
  through the squaring) — found via Gemini-surfaced competitor PR.
- **Optimizer:** Muon for the matmul parameters, AdamW for embeddings/biases.
  Muon's spectral-norm step gave a measurable BPB improvement over plain
  AdamW on the same schedule.
- **EMA:** weight EMA with decay 0.999, evaluated EMA copy, not the live one.
- **Quantization:** v2 was int8 PTQ wrapped in `.ptz` (zlib);
  v4 is **int6 QAT with LZMA compression** of the packed tensors —
  the compressed artifact fits the competition's parameter-budget rule
  while QAT keeps the accuracy loss small.

### Evaluation: the n-gram → sliding-window finding

The competition scores held-out FineWeb bytes in BPB. The default eval
chunks the validation set into non-overlapping windows. Two things I tried
on top:

1. **N-gram backoff** (α, max_order) on the model's distribution at each
   byte. Standard interpolation:
   `p_final = (1-α) · p_model + α · p_ngram`.
2. **Sliding-window eval** with stride < context_length, so every byte is
   scored under the maximum context the model has ever seen, not the
   average.

I expected (1) to help and (2) to be a minor refinement. The actual finding,
visible in [`screenshots/sweep_heatmap.png`](screenshots/):

| Eval method | Best val_bpb (seed 314, same checkpoint) |
|-------------|------------------------------------------|
| Sliding stride=64, no n-gram | **1.36165** |
| Sliding stride=128, no n-gram | 1.36179 |
| Sliding stride=256, no n-gram | 1.36238 |
| Sliding stride=64 + best n-gram (α=0.15, order=5) | 1.36345 |
| Sliding stride=64 + worst n-gram (α=0.5, order=9) | 1.42293 |

**Sliding-window evaluation dominates. Adding n-gram backoff on top makes
it strictly worse across the entire α × max_order grid I swept.**
The model has already learned the short-range statistics that the n-gram
table is approximating; mixing them in just dilutes a sharper distribution
with a fuzzier one.

This is the kind of result you only get from running the full sweep.
Reading the n-gram-backoff papers, you'd predict a small win.

### Ablation results, condensed

Full run inventory is in [`screenshots/run_audit.png`](screenshots/) — 25
ranked runs sourced from `logs/full_run_audit.txt`. The top of the table:

| Rank | val_bpb | Eval method | Steps | Seed | Run |
|------|---------|-------------|-------|------|-----|
| 1 | 1.34168 | int6 + lzma roundtrip | 2612 | 1337 | overnight_int8_best |
| 2 | 1.34988 | int8 + zlib roundtrip | 4775 | 1337 | overnight_long_ema_001 |
| 3 | 1.36165 | sliding-64 ngram=False | 1057 | 314 | v3 seed314 (sweep cache) |
| 4 | 1.36179 | sliding-128 ngram=False | 1057 | 314 | v3 seed314 (sweep cache) |
| 5 | 1.36221 | sliding-64 (R2/v3.1) | 1057 | 314 | v3.1 seed314 R2 |
| 6 | 1.36238 | sliding-256 ngram=False | 1057 | 314 | v3 seed314 (sweep cache) |
| 7 | 1.36345 | sliding-64 ngram=True  | 1057 | 314 | v3 seed314 (α=0.15, o=5) |

The submitted v4 number (1.310) reflects the int6-QAT checkpoint trained
to ~10h on a single seed under sliding-window eval; the rank-1 row above
is the closest-comparable in-tree run.

### Why DGX Spark ≠ 8×H100

The Parameter Golf reference rig is 8×H100. I ran on a single NVIDIA DGX
Spark workstation:

- **GB10 SoC, sm_121 (Blackwell)**, 128 GB UMA shared between CPU and GPU.
- **No discrete VRAM** — `nvidia-smi` reports memory as "Not Supported"
  because there is no separate device pool to report. UMA means I had
  to monitor GPU memory via `/proc/meminfo` instead.
- **PyTorch pip wheels from pytorch.org don't ship sm_121 kernels.**
  I had to use NGC CUDA 13.0 / 12.8 base images (NGC 25.12 has native
  Blackwell support; 24.12 does not) or fall back to JIT compilation.
- **CTranslate2 needed a source build for ARM64 + sm_121.**
  Some adjacent tools simply have no ARM64 Linux support
  (Wav2Lip-HQ, partial MediaPipe coverage).

Practical consequences for the competition:

1. **No 8-way data parallelism.** The leaderboard's record runs use a
   3-seed mean across 8×H100; I can do a 3-seed mean too, but each seed
   is one Spark-night, not one cluster-hour.
2. **Memory is the soft ceiling, not FLOPs.** UMA is generous (128 GB)
   but the bandwidth profile is different from a discrete H100.
   Larger batch sizes are cheap in memory and expensive in wall-clock.
3. **Quantization is more valuable than on H100.** On 8×H100 you can
   afford to keep weights in bf16; on Spark, int6 QAT is what makes the
   submission artifact fit the parameter budget at all.

The 1.310 number is therefore **not directly comparable to an 8×H100
record run.** It's the best a careful single-Spark operator could produce
in the same competition rules — which, for a portfolio piece, is
arguably the more interesting number.

---

## 4. What's in this branch

This `portfolio` branch is **not a competition submission.** It's a
snapshot of the whole project for people who didn't watch it happen:

- `PROJECT_WRITEUP.md` — this file.
- `screenshots/` — dashboard captures (see the README in there for how to
  regenerate).
- The full v4 codebase: training (`train_gpt_spark.py`,
  `training/train_int_6_qat.py`), evaluation (`eval_sliding_int6.py`,
  `eval_sliding_v31.py`, `eval_sweep.py`), the Streamlit dashboard
  (`app.py`), and the run logs under `logs/`.
- `records/`, `experiments/`, and the run-tracking DBs that backed the
  numbers above.

The actual record submission lives on `submission/v4-int6-sliding`. This
branch is based off that one, plus the writeup and screenshots.
