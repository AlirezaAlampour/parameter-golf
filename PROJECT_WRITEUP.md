# Parameter Golf: Competing in OpenAI's ML Challenge with Zero ML Background

[![Competition](https://img.shields.io/badge/OpenAI-Parameter%20Golf-412991?style=for-the-badge&logo=openai)](https://github.com/openai/parameter-golf)
[![Best Score](https://img.shields.io/badge/Best%20BPB-1.310-brightgreen?style=for-the-badge)](https://github.com/openai/parameter-golf/compare/main...AlirezaAlampour:parameter-golf:submission/v4-int6-sliding)
[![Track](https://img.shields.io/badge/Track-Non--Record%2016MB-blue?style=for-the-badge)]()
[![Hardware](https://img.shields.io/badge/Hardware-DGX%20Spark-76B900?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![AI Assisted](https://img.shields.io/badge/Built%20With-Multi--AI%20Team-ff6b6b?style=for-the-badge)]()

> **TL;DR:** I entered OpenAI's Parameter Golf competition — a challenge to build the best language model that fits in 16MB — with zero machine learning experience. I used four AI models as my engineering team. My best score: **1.310 BPB** (bits-per-byte), down from 1.666 on my first attempt. I went from not knowing what a transformer is to submitting a real competition entry in three weeks.

---

## 📊 Results Dashboard

![Progression](https://github.com/user-attachments/assets/786e153d-aaa7-4ac9-aabc-3000efd18b6e)

![Sweep Heatmaps](https://github.com/user-attachments/assets/011f326e-9df5-47ab-b6ce-bc3e33b5d2a7)

![Run Audit](https://github.com/user-attachments/assets/50aeb5c7-ff7e-4691-b7c5-128528124df4)

---

## 🏌️ The Journey

| Version | Date | BPB | Delta | What Changed |
|---------|------|-----|-------|-------------|
| v1 | Apr 7 | 1.666 | — | First ever ML model. Worse than OpenAI's baseline. |
| v2 | Apr 13 | 1.397 | −0.269 | Better hyperparameters, EMA averaging, 4hr training |
| v3 | Apr 16 | 1.368 | −0.029 | N-gram backoff cache (never submitted — obsoleted) |
| v3.1 | Apr 19 | 1.364 | −0.004 | Sliding window eval replaces n-gram |
| **v4** | **Apr 23** | **1.310** | **−0.054** | **Forgotten int6 checkpoint + sliding window** |

**Total improvement: −0.356 BPB (21% reduction from v1)**

---

## 🤖 The Multi-AI Team

The most novel aspect of this project wasn't the ML — it was the workflow. I used five AI models, each with a specific role:

| AI | Role | Why This One |
|----|------|-------------|
| **Claude** (Anthropic) | Strategy & synthesis | Best at big-picture planning, experiment design, synthesizing research from other models |
| **Claude Code** | Execution | SSH into GPU, run training, build PRs, file operations. The "hands" of the team |
| **GPT** (OpenAI) | Code & ML research | Best at writing training scripts, quantization code, deep technical ML questions |
| **Gemini** (Google) | Competitive intelligence | Long context window lets it scan 50+ competition PRs at once to extract techniques |
| **Perplexity** | Literature search | Finding academic papers with real citations on n-gram mixing, quantization, TTT |
| ~~**Grok**~~ | ~~Research~~ | **Fired.** Reported 0.0000 BPB (physically impossible perfect compression). Caught hallucinating. |

### How it works
Me: "What should we try next?"
→ Claude: designs experiment, writes research prompts
→ I paste prompts into GPT, Gemini, Perplexity
→ They return findings
→ I paste results back to Claude for synthesis
→ Claude writes implementation plan
→ Claude Code executes on GPU
→ Results feed back into the loop

I'm the project manager. The AIs are the team.

---

## 🔬 Key Discoveries

### 1. The 3am Sweep
Designed a 27-cell overnight experiment testing n-gram cache vs sliding window evaluation. Went to bed expecting n-gram to win. Woke up to find sliding window dominates — three days of n-gram work thrown away. That's the scientific method.

### 2. The Forgotten Checkpoint
Hours before submitting v3.1, ran a full audit of every training run on disk. Found a checkpoint from April 11 that was **0.05 BPB better** than everything I'd spent the past week building. Almost submitted the wrong model. The AI audit caught it.

### 3. The Camping Stove Problem
My DGX Spark trains 1,040 steps in 4 hours. Competition 8×H100 hardware trains 5,000-10,000 steps in 10 minutes. I'm not losing on technique — I'm losing on hardware. Every trick the leaders use, I have too. They just cook 10× longer.

### 4. The Grok Incident
Asked Grok to check the leaderboard. It reported the #1 score as 0.0000 BPB — literally perfect prediction of all human language. Physically impossible. Asked again. It doubled down. Fired immediately.

---

## 🏗️ Architecture
7-Layer UNet Shared Transformer
├── d=512, 8 heads (4 KV heads, GQA)
├── MLP 4× with LeakyReLU² (negative_slope=0.5)
├── Int6 per-row QAT + LZMA compression → 13.73 MB artifact
├── EMA averaging (decay=0.997)
├── Muon optimizer + Adam for embeddings
├── RoPE positional encoding
├── Tied embeddings, SentencePiece BPE (vocab 1024)
└── Sliding window eval (stride=64, 969K windows)

**20.7M parameters → 13.73 MB artifact (2.27 MB under 16 MB cap)**

---

## 📈 Ablation: Sliding Window vs N-gram Cache

The overnight sweep that killed three days of work:

| Stride | N-gram | BPB | Delta vs baseline |
|--------|--------|-----|------------------|
| 64 | OFF | **1.36165** | **−0.00920** |
| 64 | ON (α=0.15) | 1.36345 | −0.00740 |
| 128 | OFF | 1.36179 | −0.00906 |
| 256 | OFF | 1.36238 | −0.00847 |

**Sliding window alone beats every n-gram configuration.** N-gram mixing is net harmful at stride=64 because the model with full context already captures what the cache provides.

---

## 🏆 Competition Context

| Entry | BPB | Gap to Our Best |
|-------|-----|----------------|
| Record SOTA | ~1.03 | −0.28 |
| Non-record leaders | ~1.12 | −0.19 |
| **Our v4** | **1.310** | **—** |
| OpenAI baseline | 1.224 | +0.086 |
| Our v1 (first attempt) | 1.666 | +0.356 |

The gap to leaders is primarily compute, not technique. See "The Camping Stove Problem" above.

---

## 📁 Repository Structure
├── records/track_non_record_16mb/
│   └── 2026-04-23_SharedTransformer_Int6_SlidingWindow_7L/
│       ├── train_gpt.py          # Self-contained training + eval
│       ├── submission.json       # Competition manifest
│       ├── README.md             # Technical submission writeup
│       └── logs/                 # Training, eval, and ablation logs
├── PROJECT_WRITEUP.md            # This file
└── screenshots/                  # Dashboard captures

---

## 🔗 Links

- **Competition:** [openai/parameter-golf](https://github.com/openai/parameter-golf)
- **My v4 Submission PR:** [View on GitHub](https://github.com/openai/parameter-golf/compare/main...AlirezaAlampour:parameter-golf:submission/v4-int6-sliding)
- **YouTube Video:** *coming soon*

---

## 🎓 What I Learned

1. **AI models are better as a team than solo.** Claude couldn't code well. GPT couldn't strategize. Gemini fabricated details. The combination worked.
2. **Scientific method > intuition.** Every confident prediction from an AI model was wrong at least once. The overnight sweep that killed n-gram proved: measure, don't assume.
3. **The human's job is judgment.** When to stop optimizing. When to ship. When to throw away three days of work. Those decisions were mine.
4. **Multi-agent workflows are the future.** The workflow I built — delegating research, synthesizing, designing experiments, executing — will be normal in 3 years. I just did it early.

---

*Built in 3 weeks with zero ML background. Powered by Claude, GPT, Gemini, Perplexity, and one fired Grok.*
