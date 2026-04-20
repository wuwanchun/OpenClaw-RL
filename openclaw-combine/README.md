# Combined Binary RL + On-Policy Distillation

*Let us build on each other's strengths and offset each other's weaknesses.*

This method runs Binary RL (GRPO) and On-Policy Distillation (OPD) simultaneously, combining evaluative and directional signals into a single training objective. In our experiments, this achieves significant performance gains over either method alone.

## Why Combine?

| Dimension | Binary RL | OPD | Combined |
|---|---|---|---|
| Signal type | Evaluative (good/bad) | Directional | Evaluative + directional |
| Advantage | Sequence-level scalar | Token-level directional | Mixed sequence and token-level |
| Density | All scored turns | Hint-accepted turns only | All scored turns |
| Feedback type | User / environment | Explicit corrections | Both implicit and explicit feedback |
| Signal richness | 1 scalar per sample | 1 value per token | 1 value per token |

Binary RL accepts every scored turn, requires no hint extraction, and works with any next-state signal — including terse, implicit reactions (a user simply re-asking a question) or structured environment outputs (exit codes, test verdicts). OPD should be enabled additionally when the interaction stream is likely to carry rich directive content: users who give explicit corrections ("don't use that library", "check the file first"), or environments that produce detailed error traces (SWE diffs, compiler diagnostics).

In practice, Binary RL provides broad gradient coverage across all turns, while OPD provides high-resolution, per-token corrections on the subset of turns where directive signals are available.

## Combined Advantage

Both branches share the same PPO clipped surrogate loss — only the advantage computation differs. The combined advantage is:

$$A_t = w_{\text{binary}} \, r_{\text{final}} + w_{\text{opd}} \left( \log \pi_{\text{teacher}}(a_t \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t \mid s_t) \right)$$

where $w_{\text{binary}} = w_{\text{opd}} = 1$ by default. OPD samples carry `reward=0` so their GRPO advantage is zero; RL samples carry `teacher_logp ≈ rollout_logp` so their teacher advantage is approximately zero. Each branch naturally dominates for its own sample type, and the combined advantage is simply their sum.

## Per-Turn Pipeline

For each main-line turn, after the next state arrives:

1. Run `m` hint-judge votes and `m` eval votes concurrently.
2. If the hint is accepted (longest non-trivial positive hint), emit one **OPD** sample with teacher log-probs.
3. If the eval score is `+1` or `−1`, emit one **RL** sample with the scalar reward.
4. A single turn can contribute both sample types.
5. Training batch fires when collected sample count reaches `rollout_batch_size`.

## How to Run

```bash
cd slime
# Qwen3
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine.sh
```

### Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENCLAW_COMBINE_W_RL` | `1.0` | Weight $w_{\text{binary}}$ for the GRPO advantage |
| `OPENCLAW_COMBINE_W_OPD` | `1.0` | Weight $w_{\text{opd}}$ for the teacher advantage |
| `PRM_M` | `1` | Number of independent judge/eval votes per turn |

All other variables (`NUM_GPUS`, `ACTOR_GPUS`, `HF_CKPT`, etc.) are shared with the Binary RL and OPD scripts — see the [main README](../README.md) for the full list.

## File Structure

```text
openclaw-combine/
├── README.md
├── run_qwen3_4b_openclaw_combine.sh          # Launch script (Qwen3)
├── run_qwen35_4b_openclaw_combine.sh         # Launch script (Qwen3.5)
├── openclaw_combine_api_server.py            # Async proxy: hint judge + PRM eval + sample submission
├── openclaw_combine_rollout.py               # Rollout bridge to SLIME trainer
├── combine_loss.py                           # Weighted advantage: w_rl * GRPO + w_opd * teacher
└── results/                                  # Runtime records (auto-created)
```

---

## Single-GPU INT4 QLoRA

Train and serve on a **single 24 GB GPU** using `run_qwen3_4b_openclaw_combine_single_gpu_int4_qlora.sh`.

### Required Environment Variables

| Variable | Description |
|---|---|
| `HF_CKPT` | Path to original Qwen3-4B HF weights (**not** pre-quantized) |
| `OPENAI_API_KEY` | API key for external LLM (used by PRM and OPD teacher) |
| `OPENAI_BASE_URL` | Base URL for the external LLM API |
| `EXTERNAL_MODEL` | Model name served by the external API |

### Optional Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MICRO_BATCH_SIZE` | `1` | Micro-batch size per GPU |
| `ROLLOUT_BATCH_SIZE` | `16` | Samples collected before each training step |

### Key Features Enabled

| Feature | Flag / Config | Description |
|---|---|---|
| INT4 quantization | `--fsdp-load-in-4bit` | bitsandbytes NF4 quantization. Compresses base model weights to ~4 bits, reducing VRAM from ~8 GB (bf16) to ~2 GB for Qwen3-4B. Only LoRA adapters are trained in full precision. |
| LoRA fine-tuning | `--use-lora --lora-rank 4` | Low-Rank Adaptation. Freezes base weights and inserts small trainable matrices into attention/MLP layers.
| Gradient checkpointing | `--gradient-checkpointing` | Trades compute for memory by recomputing activations during backward pass instead of storing them. |
| Colocate (train + rollout on same GPU) | `--colocate` | Runs both the FSDP training actor and the SGLang rollout engine on the same GPU(s). Essential for single-GPU setups. |
| Rollout offload | `--offload-rollout` | When training starts, the SGLang rollout engine is offloaded from GPU to free VRAM for the training forward/backward pass. When training finishes, the engine is reloaded for the next rollout. This alternating pattern allows a single GPU to handle both roles. |
| External PRM API | `--prm-use-external-api` | Sends PRM scoring requests to an external OpenAI-compatible API instead of hosting a local PRM model on GPU. Eliminates the need for a dedicated PRM GPU — critical for single-GPU setups. Configured via `PRM_EXTERNAL_API_BASE`, `PRM_EXTERNAL_MODEL`, and `PRM_EXTERNAL_API_KEY`. |
| Train sequence truncation | `SLIME_TRAIN_MAX_SEQ_LEN=4096` | Caps the token length of each training sample. Sequences longer than this are truncated from the **left** (dropping early prompt tokens, preserving the response). Prevents OOM on long rollouts and keeps per-sample memory bounded. |
| Logit chunking | `SLIME_LOGIT_CHUNK_SIZE=512` | Computes log-softmax and entropy in chunks of this size instead of materializing the full `[seq_len, vocab_size]` tensor at once. For Qwen3-4B (vocab ~152 K), this reduces peak allocation from ~7.5 GB to ~0.9 GB per sample. Set to `0` to disable chunking. |

### Quick Start

```bash
cd slime

export HF_CKPT="/path/to/Qwen3-4B"
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-api-base"
export EXTERNAL_MODEL="your-model-name"

bash ../openclaw-combine/run_qwen3_4b_openclaw_combine_single_gpu_int4_qlora.sh
```

The model is served at `http://0.0.0.0:30000/v1` by default. For single-GPU evaluation scripts, see [`../openclaw-test/README.md`](../openclaw-test/README.md).
