#!/usr/bin/env bash
# 一键闭环：切分 -> 收集 -> 进化 -> (可选训练) -> 评测 -> 汇总
# 环境变量见各步骤注释；最少需要 WCB_ROOT + 模型服务已启动。
set -euo pipefail

ABLATION_ROOT="${ABLATION_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"
WCB_ROOT="${WCB_ROOT:?set WCB_ROOT to the WildClawBench clone}"
REPO_ROOT="$(cd -- "${ABLATION_ROOT}/.." &>/dev/null && pwd)"

# 模型名不设默认：留空时由 run_variant.sh 按 SIZE 解析
BASE_MODEL="${BASE_MODEL:-}"
RL_MODEL="${RL_MODEL:-}"
ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-4}"
DO_TRAIN="${DO_TRAIN:-0}"   # 1 = 跑离线 GRPO
MODE="${MODE:-full}"        # full = 四组消融; skill_only = 只跑 skill 进化线
SIZE="${SIZE:-0p6b}"         # 0p6b|4b|8b，传给 run_variant.sh

cd "${WCB_ROOT}"

echo "== [0/6] preflight: docker image =="
DOCKER_IMAGE_NAME="${DOCKER_IMAGE:-wildclawbench-ubuntu:v1.3}"
if ! docker image inspect "${DOCKER_IMAGE_NAME}" >/dev/null 2>&1; then
  echo "[cycle] ERROR: image '${DOCKER_IMAGE_NAME}' not found locally." >&2
  echo "  It is NOT on Docker Hub. Load it from the HF tarball first:" >&2
  echo "    hf download internlm/WildClawBench Images/wildclawbench-ubuntu_v1.3.tar --repo-type dataset --local-dir ." >&2
  echo "    docker load -i Images/wildclawbench-ubuntu_v1.3.tar" >&2
  exit 1
fi

# skill-only 模式: 不训练, 只评 base / skill_only
VARIANTS="base skill_only rl_only rl_skill"
if [[ "${MODE}" == "skill_only" ]]; then
  DO_TRAIN=0
  VARIANTS="base skill_only"
  echo "[cycle] MODE=skill_only: skip training, eval {base, skill_only}"
fi

echo "== [1/6] split =="
python3 "${ABLATION_ROOT}/scripts/make_split.py" \
  --wcb-root "${WCB_ROOT}" \
  ${WCB_CATEGORIES:+--categories ${WCB_CATEGORIES}} \
  --output "${ABLATION_ROOT}/configs/split.json"

echo "== [2/6] collect train-split trajectories (base model) =="
case "${SIZE}" in
  0p6b) SIZE_NAME="qwen3-0p6b" ;;
  4b)   SIZE_NAME="qwen3-4b-instruct" ;;
  8b)   SIZE_NAME="qwen3-8b" ;;
  *) echo "unknown SIZE: ${SIZE} (0p6b|4b|8b)" >&2; exit 2 ;;
esac
COLLECT_MODEL="${BASE_MODEL:-${SIZE_NAME}-base}"
RUN_NAME=collect_base ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK}" \
  bash "${ABLATION_ROOT}/scripts/run_tasks.sh" train "local/${COLLECT_MODEL}" 0

echo "== [3/6] skill evolution round =="
# 进化轮数计数：skills/.evolve_epoch 持久化，每轮 +1；报告按轮数留档
EPOCH_FILE="${ABLATION_ROOT}/skills/.evolve_epoch"
EVOLVE_EPOCH=$(( $(cat "${EPOCH_FILE}" 2>/dev/null || echo 0) + 1 ))
export EVOLVE_EPOCH
# 进化器 LLM 默认复用本地 sglang 服务；不配则只剩启发式兜底（易全 skip）
export SKILL_LLM_API_BASE="${SKILL_LLM_API_BASE:-http://127.0.0.1:${PORT:-8000}/v1}"
export SKILL_LLM_MODEL="${SKILL_LLM_MODEL:-${COLLECT_MODEL}}"
echo "[cycle] evolver LLM: ${SKILL_LLM_API_BASE} model=${SKILL_LLM_MODEL} epoch=${EVOLVE_EPOCH}"

PYTHONPATH="${REPO_ROOT}/slime-coding-agent${PYTHONPATH:+:${PYTHONPATH}}" \
python3 -m skill_evolve.run_round \
  --raw-dir "${ABLATION_ROOT}/results/collect_base/raw" \
  --skills-dir "${ABLATION_ROOT}/skills" \
  --report "${ABLATION_ROOT}/results/evolve_report.json"
echo "${EVOLVE_EPOCH}" > "${EPOCH_FILE}"
cp "${ABLATION_ROOT}/results/evolve_report.json" \
   "${ABLATION_ROOT}/results/evolve_report_e${EVOLVE_EPOCH}.json"

if [[ "${DO_TRAIN}" == "1" ]]; then
  echo "== [4/6] offline GRPO training =="
  python3 "${ABLATION_ROOT}/scripts/wcb_to_rl_dataset.py" \
    --raw-dir "${ABLATION_ROOT}/results/collect_base/raw" \
    --model "${HF_CKPT:?HF_CKPT required for training}" \
    --output "${ABLATION_ROOT}/results/rl_data/{rollout_id}.pt" \
    --reward-mode raw
  RL_DATA="${ABLATION_ROOT}/results/rl_data/{rollout_id}.pt" \
    bash "${ABLATION_ROOT}/scripts/train_grpo_offline.sh"
else
  echo "== [4/6] skip training (DO_TRAIN=0) =="
fi

echo "== [5/6] evaluate 4 variants on held-out eval split =="
SPLIT_HASH="$(md5sum "${ABLATION_ROOT}/configs/split.json" | cut -d' ' -f1)"
for variant in ${VARIANTS}; do
  # base 评测缓存：切分+尺寸没变就复用上轮结果（REUSE_BASE_EVAL=0 强制重跑）
  if [[ "${variant}" == "base" && "${REUSE_BASE_EVAL:-1}" == "1" ]]; then
    DONE_FILE="${ABLATION_ROOT}/results/base/.eval_done"
    if [[ -f "${DONE_FILE}" ]] \
       && grep -q "^split=${SPLIT_HASH}$" "${DONE_FILE}" \
       && grep -q "^size=${SIZE}$" "${DONE_FILE}"; then
      echo "[cycle] base eval 命中缓存（split+size 未变），跳过 -> results/base"
      continue
    fi
  fi
  SIZE="${SIZE}" BASE_MODEL="${BASE_MODEL}" RL_MODEL="${RL_MODEL}" \
    bash "${ABLATION_ROOT}/scripts/run_variant.sh" "${variant}"
  if [[ "${variant}" == "base" ]]; then
    printf 'split=%s\nsize=%s\ndate=%s\n' "${SPLIT_HASH}" "${SIZE}" "$(date -Iseconds)" \
      > "${ABLATION_ROOT}/results/base/.eval_done"
  fi
done

echo "== [6/6] compare =="
python3 "${ABLATION_ROOT}/scripts/compare_results.py" \
  --results-dir "${ABLATION_ROOT}/results" \
  --output "${ABLATION_ROOT}/results/ablation_summary.json"

echo "done. summary: ${ABLATION_ROOT}/results/ablation_summary.json"
