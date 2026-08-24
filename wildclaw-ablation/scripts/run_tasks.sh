#!/usr/bin/env bash
# 按任务清单逐个跑 WildClawBench（run_batch.py 单次只接一个 --task）。
# 用法: bash run_tasks.sh <train|eval|任务列表文件|单个task.md> <model_id> [inject_skills 0|1]
# ROLLOUTS_PER_TASK: 每个任务重复次数（GRPO 组内需要 >1 条 rollout）
set -euo pipefail

SPLIT_KEY="${1:?train|eval|path-to-list}"
MODEL_ID="${2:?model id, e.g. local/qwen3-0p6b-base}"
INJECT_SKILLS="${3:-0}"

ABLATION_ROOT="${ABLATION_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"
WCB_ROOT="${WCB_ROOT:?set WCB_ROOT}"
REPO_ROOT="$(cd -- "${ABLATION_ROOT}/.." &>/dev/null && pwd)"
SPLIT_JSON="${SPLIT_JSON:-${ABLATION_ROOT}/configs/split.json}"
MODELS_CONFIG="${MODELS_CONFIG:-${ABLATION_ROOT}/configs/my_api.json}"
SKILLS_SRC="${SKILLS_SRC:-${ABLATION_ROOT}/skills}"
RUN_NAME="${RUN_NAME:-run}"
RESULTS_DIR="${ABLATION_ROOT}/results/${RUN_NAME}"
ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-1}"

# ---- resolve task list ----
if [[ "${SPLIT_KEY}" == "train" || "${SPLIT_KEY}" == "eval" ]]; then
  LIST_FILE="${RESULTS_DIR}/.tasks_${SPLIT_KEY}.txt"
  mkdir -p "${RESULTS_DIR}"
  python3 - "${SPLIT_JSON}" "${SPLIT_KEY}" > "${LIST_FILE}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print("\n".join(data[sys.argv[2]]))
PY
elif [[ "${SPLIT_KEY}" == *.md ]]; then
  LIST_FILE="${RESULTS_DIR}/.tasks_single.txt"
  mkdir -p "${RESULTS_DIR}"
  printf '%s\n' "${SPLIT_KEY}" > "${LIST_FILE}"
else
  LIST_FILE="${SPLIT_KEY}"
fi

# ---- lobster workspace (skill variants) ----
# 注意：注入后落在容器 /root/ 下；openclaw 2026.3.11 实测只发现
# /root/.openclaw/skills（managed），不发现 /root/skills（workspace），
# 所以技能要放在 .openclaw/skills/ 子路径。
LOBSTER_ARGS=()
if [[ "${INJECT_SKILLS}" == "1" ]]; then
  WORKSPACE_DIR="${RESULTS_DIR}/lobster_workspace"
  rm -rf "${WORKSPACE_DIR}"
  mkdir -p "${WORKSPACE_DIR}/.openclaw/skills"
  count=0
  if [[ -d "${SKILLS_SRC}" ]]; then
    # EvolvingSkillStore 布局: <skills_root>/<name>/SKILL.md
    for d in "${SKILLS_SRC}"/*/; do
      [[ -f "${d}SKILL.md" ]] || continue
      name="$(basename "${d}")"
      mkdir -p "${WORKSPACE_DIR}/.openclaw/skills/${name}"
      cp "${d}SKILL.md" "${WORKSPACE_DIR}/.openclaw/skills/${name}/SKILL.md"
      count=$((count + 1))
    done
  fi
  echo "[run_tasks] skills injected: ${count}"
  # run_batch 要求 --lobster-name 与 --lobster-workspace 成对（缺失会 sys.exit(1)）；
  # 用 RUN_NAME 当名字，输出目录自动带变体前缀
  LOBSTER_ARGS=(--lobster-workspace "${WORKSPACE_DIR}" --lobster-name "${RUN_NAME}")
fi

mkdir -p "${RESULTS_DIR}/raw"
# 增量拷贝标记：output/ 会累积所有历史 run，逐任务全量 cp 是平方级膨胀，
# 只拷 marker 之后新增的 run 目录（output/openclaw/<cat>/<task>/<run_id>）
MARKER="${RESULTS_DIR}/.copy_marker"
touch "${MARKER}"
cd "${WCB_ROOT}"

total=0
for ((rep=1; rep<=ROLLOUTS_PER_TASK; rep++)); do
  while IFS= read -r task; do
    [[ -n "${task}" ]] || continue
    total=$((total + 1))
    echo "[run_tasks] (${total}, rep ${rep}/${ROLLOUTS_PER_TASK}) ${task}"
    # run_batch 在任务评分带 error（如 0 分）时也 sys.exit(1)；
    # set -e 下不能让单个任务的失败终止整个批次 —— 失败轨迹同样是进化数据
    python3 eval/run_batch.py \
      --task "${task}" \
      --models-config "${MODELS_CONFIG}" \
      --model "${MODEL_ID}" \
      "${LOBSTER_ARGS[@]}" || echo "[run_tasks] WARN: non-zero exit for ${task}, continuing"
    if [[ -d output ]]; then
      while IFS= read -r d; do
        [[ -n "${d}" ]] || continue
        mkdir -p "${RESULTS_DIR}/raw/$(dirname "${d}")"
        cp -r "output/${d}" "${RESULTS_DIR}/raw/${d}"
      done < <(cd output && find . -mindepth 4 -maxdepth 4 -type d -newer "${MARKER}" | sed 's|^\./||')
      touch "${MARKER}"
    fi
  done < "${LIST_FILE}"
done

if [[ -f output/summary_all.json ]]; then
  cp output/summary_all.json "${RESULTS_DIR}/summary_all.json"
fi
echo "[run_tasks] done: ${total} task(s) -> ${RESULTS_DIR}"
