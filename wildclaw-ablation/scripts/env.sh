#!/usr/bin/env bash
# 共享环境：两个 SSH 窗口的启动脚本都 source 这个文件。
# 路径按服务器布局写死默认值，可用环境变量覆盖。

# ---- 目录布局（服务器） ----
# ~/Desktop/OpenClaw-RL            仓库（分支 codex/create-self-evolving-coding-agent）
#   └─ wildclaw-ablation/          ABLATION_ROOT: scripts/ configs/ results/ skills/
#   └─ slime-coding-agent/         skill_evolve 进化器包
#   └─ models/Qwen3-8B 等          模型权重
# ~/Desktop/WildClawBench          WCB_ROOT: eval/ workspace/ output/ .venv/

export ABLATION_ROOT="${ABLATION_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"
export REPO_ROOT="$(cd -- "${ABLATION_ROOT}/.." &>/dev/null && pwd)"
export WCB_ROOT="${WCB_ROOT:-${HOME}/Desktop/WildClawBench}"

# ---- 模型服务 ----
export PORT="${PORT:-8000}"
# 循环检测镜像（推荐）：由 v1.3 派生，唯一区别是 openclaw.json 开了
# tools.loopDetection（warning=2 / critical=3 / breaker=5，即重复 5 轮直接熔断），
# 把死循环 run 提前终止。构建方法见 README「循环检测」一节；没构建过就保持 v1.3。
export DOCKER_IMAGE="${DOCKER_IMAGE:-wildclawbench-ubuntu:v1.3-loopguard}"

# ---- 消融运行参数（run_cycle.sh 用） ----
export SIZE="${SIZE:-8b}"                         # 0p6b|4b|8b
export MODE="${MODE:-skill_only}"                 # skill_only|full
export ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-1}"  # 判分制下 1 次就够；GRPO 训练时需要 >=4 凑组
export JOBS="${JOBS:-1}"                              # 任务并发数；8B@5090 建议 2-3
export REUSE_BASE_EVAL="${REUSE_BASE_EVAL:-1}"        # 1 = base eval 只跑一次，切分+尺寸没变就复用

# ---- skill 进化器 LLM（默认复用本地 sglang；run_cycle.sh 里也有兜底） ----
export SKILL_LLM_API_BASE="${SKILL_LLM_API_BASE:-http://127.0.0.1:${PORT}/v1}"
# SKILL_LLM_MODEL 由 run_cycle.sh 按 SIZE 推导（如 qwen3-8b-base），这里不设死

# ---- 搜索类任务（04_Search_Retrieval）需要真 Brave key 才能加回 ----
# export WCB_CATEGORIES="01_Productivity_Flow 03_Social_Interaction 04_Search_Retrieval 06_Safety_Alignment"
# export BRAVE_API_KEY="<real key>"

echo "[env] ABLATION_ROOT=${ABLATION_ROOT}"
echo "[env] WCB_ROOT=${WCB_ROOT} SIZE=${SIZE} MODE=${MODE} ROLLOUTS=${ROLLOUTS_PER_TASK} PORT=${PORT}"
