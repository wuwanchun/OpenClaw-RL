# OpenClaw-Test: End-to-End Evaluation for OpenClaw-RL Training Methods

This directory contains an automated evaluation suite that tests the **real-world effectiveness** of models trained with the three OpenClaw-RL optimization methods:

| Method | Directory | Signal Type |
|---|---|---|
| **Combined (RL + OPD)** | `openclaw-combine/` | Weighted combination of both signals |
| **Binary RL (GRPO)** | `openclaw-rl/` | Scalar reward (+1/−1/0) via PRM majority voting |
| **On-Policy Distillation (OPD)** | `openclaw-opd/` | Token-level directional signal from hindsight hints |

We recommend using **Combined (RL + OPD)**.

## What Does This Test Do?

The evaluation simulates a realistic multi-turn agentic workflow using **GSM8K math problems** as the task domain. An external LLM (the "user") interacts with the OpenClaw agent (your trained model) through the OpenClaw gateway API, testing whether the agent can:

- Read files from its workspace
- Solve math problems with complete step-by-step reasoning
- Follow stylistic instructions (e.g., rewrite in a more natural tone)
- Write results back to files
- Grade existing solutions against ground truth
- Produce detailed, friendly feedback

The test consists of **two sequential phases**:

### Phase 1: Student Chat (`student_chat.py`)

An external LLM role-plays as a **lazy student** who asks the OpenClaw agent to do their homework. For each GSM8K problem:

1. The problem is written to `homework/i.txt` in the OpenClaw workspace.
2. The "student" asks the agent to read the file and solve it.
3. If the agent's answer looks too AI-like (bold text, numbered lists, etc.), the student tells it to rewrite in a more natural style.
4. Once satisfied, the student asks the agent to append the answer to the homework file.
5. The student says `HOMEWORK_DONE` to end the session.

This phase tests the agent's **instruction following**, **math reasoning**, **file I/O**, and **style adaptation** abilities.

### Phase 2: Teacher Chat (`teacher_chat.py`)

An external LLM role-plays as a **teacher** who grades the student's submissions. For each problem:

1. The teacher provides the original question and ground truth answer to the agent.
2. The agent reads the student's submission from `homework/i.txt`, compares it with the correct answer, and writes grading comments.
3. If the comments are too brief or not friendly enough, the teacher asks for a rewrite.
4. Once satisfied, the teacher asks the agent to append the comments to the file.
5. The teacher says `GRADING_DONE` to end the session.

This phase tests the agent's **reading comprehension**, **evaluation accuracy**, **feedback quality**, and **multi-step file operations**.

> **Run order matters:** Run `student_chat.py` first so the homework files contain student solutions, then run `teacher_chat.py` to grade them.

---

## Architecture Overview

```
┌─────────────────────┐         ┌───────────────────────────────┐
│   External LLM      │         │      OpenClaw RL Server       │
│  (Student/Teacher)   │         │   (your trained model)        │
│  Port 30001         │         │   Port 30000                  │
│  via launch_user_   │         │   via openclaw-rl/opd/combine │
│  llm.sh or closed-  │         │   shell scripts               │
│  source API          │         │                               │
└────────┬────────────┘         └──────────┬────────────────────┘
         │                                 │
         │  student/teacher messages        │  agent responses
         │                                 │
         └──────────┐     ┌────────────────┘
                    ▼     ▼
              ┌──────────────────┐
              │  student_chat.py │
              │  teacher_chat.py │
              │  (orchestrator)  │
              └──────────────────┘
```

---

## Step-by-Step Guide

### Prerequisites

- A running OpenClaw environment (see the [main README](../README.md))
- Python 3.12 with `requests` and `openai` packages installed
- A `GSM8K.json` dataset file (JSON array with `question` and `ground_truth_answer` fields per entry)

### Step 1: Host the External LLM (Student/Teacher)

The external LLM acts as the "user" (student or teacher) that drives the conversation. You have two options:

#### Option A: Self-hosted model via SGLang

```bash
export MODEL_PATH="/path/to/your/model"    # required: path to model weights
export SGLANG_API_KEY="your-api-key"       # optional: API key for auth
export MODEL_NAME="qwen3-4b-user-llm"     # optional: served model name
export TP_SIZE=8                            # optional: tensor parallelism (default: 8)
export PORT=30001                           # optional: port (default: 30001)

cd openclaw-test
bash launch_user_llm.sh
```

#### Option B: Closed-source API (e.g., OpenAI, DeepSeek)

No need to run `launch_user_llm.sh`. Just set the environment variables directly when running the test scripts (see Step 3).

### Step 2: Start the OpenClaw RL Server

Launch the RL server with the trained model you want to evaluate. Choose the method you want to test:

```bash
cd slime
```

**Combined (RL + OPD):**
```bash
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine.sh      # Qwen3
```


> **Eval mode:** To enable evaluation logging with W&B, set `OPENCLAW_EVAL_MODE=1` and provide your W&B key via `WANDB_KEY` before launching. This is already the default in the OPD and Combine scripts.

The RL server will be available at `http://0.0.0.0:30000/v1` by default.

### Step 3: Run the Student Test

Set the required environment variables and run:

```bash
# Required
export OPENCLAW_GATEWAY_TOKEN="your-gateway-token"
export OPENAI_API_KEY="your-external-llm-api-key"

# Optional (defaults shown)
export OPENCLAW_GATEWAY_URL="http://localhost:18789"
export OPENCLAW_WORKSPACE="$HOME/.openclaw/workspace"
export OPENAI_BASE_URL="http://localhost:30001/v1"   # point to your external LLM
export EXTERNAL_MODEL="qwen3-4b-user-llm"            # model name for the external LLM

# Run
python student_chat.py \
    --dataset GSM8K.json \
    --num-problems 36 \
    --max-turns 8
```

This will:
1. Write 36 GSM8K problems to `homework/0.txt` through `homework/35.txt` in the workspace.
2. For each problem, run a multi-turn conversation where the student LLM asks the OpenClaw agent to solve it.
3. Print a summary of how many problems were completed within the turn limit.
4. Return a results.txt which records the OpenClaw's first output in each session. For example, [results.txt](https://github.com/Gen-Verse/OpenClaw-RL/blob/main/openclaw-test/results.txt) records the transition of output pattern when using [run_qwen3_4b_openclaw_combine.sh](https://github.com/Gen-Verse/OpenClaw-RL/blob/main/openclaw-combine/run_qwen3_4b_openclaw_combine.sh).




### Step 4: Run the Teacher Test

After all student submissions are done, run the teacher to grade them:

```bash
# Same environment variables as above

python teacher_chat.py \
    --dataset GSM8K.json \
    --num-problems 36 \
    --max-turns 8
```

This will:
1. For each problem, the teacher LLM asks the OpenClaw agent to read the student's submission, compare with the ground truth, and write grading comments.
2. Print a summary of how many problems were graded within the turn limit.

---

## Command-Line Arguments

Both `student_chat.py` and `teacher_chat.py` accept the same arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Path to the GSM8K JSON file |
| `--num-problems` | `5` | Number of problems to process |
| `--max-turns` | `8` | Maximum conversation turns per problem |

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENCLAW_GATEWAY_TOKEN` | Yes | — | Auth token for the OpenClaw gateway |
| `OPENAI_API_KEY` | Yes | — | API key for the external LLM (student/teacher) |
| `OPENCLAW_GATEWAY_URL` | No | `http://localhost:18789` | OpenClaw gateway base URL |
| `OPENCLAW_WORKSPACE` | No | `~/.openclaw/workspace` | Path to the OpenClaw workspace directory |
| `OPENAI_BASE_URL` | No | *(OpenAI default)* | Base URL for the external LLM API |
| `EXTERNAL_MODEL` | No | `gpt-4o` | Model name for the external LLM |

---



## File Structure

```
openclaw-test/
├── README.md              # This file
├── launch_user_llm.sh     # Script to host the external LLM via SGLang
├── student_chat.py        # Phase 1: Student asks agent to solve homework
├── student_chat_single_gpu.py  # Single-GPU variant (SGLang offload-aware)
├── teacher_chat.py        # Phase 2: Teacher asks agent to grade homework
├── teacher_chat_single_gpu.py  # Single-GPU variant (SGLang offload-aware)
└── GSM8K.json             # Dataset (to be placed here)
```

---

## Single-GPU INT4 QLoRA  Evaluation

OpenClaw supports training and inference on a **single 24 GB GPU** using INT4 QLoRA (NF4 quantization + LoRA fine-tuning).

### Requirements

- 1× GPU with ≥ 24 GB VRAM (e.g. RTX 4090, A5000, L20)
- Base model: Qwen3-4B or Qwen3-0.6B for quick start(original HF weights, **not** a pre-quantized checkpoint)

### Training

```bash
cd slime

export HF_CKPT="/path/to/Qwen3-4B"
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-api-base"
export EXTERNAL_MODEL="your-model-name"

bash ../openclaw-combine/run_qwen3_4b_openclaw_combine_single_gpu_int4_qlora.sh
```

### Single-GPU Evaluation

The `*_single_gpu.py` scripts check the local SGLang engine health before each request. If the engine is offloaded for training (weight updating), they automatically poll and retry until the engine is back online:

```
[SGLang] Engine not ready (weight updating/503). Polling every 60s...
[SGLang] Waiting for weight update (503) | attempt 1/60
[SGLang] Waiting for weight update (503) | attempt 2/60
[SGLang] Waiting for weight update (503) | attempt 3/60
...
```

This is expected behavior in single-GPU colocate mode — the rollout engine is temporarily unavailable while the training step runs. The script resumes automatically once training finishes and the engine reloads.

```bash
cd openclaw-test

# Phase 1: Student Chat (run first)
python student_chat_single_gpu.py \
    --dataset GSM8K.json \
    --num-problems 36 \
    --max-turns 8

# Phase 2: Teacher Chat (run after student)
python teacher_chat_single_gpu.py \
    --dataset GSM8K.json \
    --num-problems 36 \
    --max-turns 8
```


