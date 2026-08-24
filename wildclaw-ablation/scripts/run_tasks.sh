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
JOBS="${JOBS:-1}"   # 并发任务数；>1 时同 rep 内任务并行（sglang 8B 扛 2-3 路没问题）

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

# 版本戳：每批 run 记录代码/镜像/并发配置，之后可按版本分组统计
GIT_REV="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
WCB_STATE="$(git -C "${WCB_ROOT}" diff --quiet 2>/dev/null && echo clean || echo dirty)"
cat > "${RESULTS_DIR}/run_info.json" <<EOF
{
  "git_rev": "${GIT_REV}",
  "wcb_state": "${WCB_STATE}",
  "docker_image": "${DOCKER_IMAGE:-unknown}",
  "jobs": ${JOBS},
  "rollouts_per_task": ${ROLLOUTS_PER_TASK},
  "model_id": "${MODEL_ID}",
  "inject_skills": ${INJECT_SKILLS},
  "started_at": "$(date -Iseconds)"
}
EOF
echo "[run_tasks] run_info: git=${GIT_REV} image=${DOCKER_IMAGE:-unknown} jobs=${JOBS} (wcb ${WCB_STATE})"

# 增量拷贝标记：output/ 会累积所有历史 run，逐任务全量 cp 是平方级膨胀，
# 只拷 marker 之后新增的 run 目录（output/openclaw/<cat>/<task>/<run_id>）
MARKER="${RESULTS_DIR}/.copy_marker"
touch "${MARKER}"
cd "${WCB_ROOT}"

run_one() {
  local task="$1"
  # run_batch 在任务评分带 error（如 0 分）时也 sys.exit(1)；
  # set -e 下不能让单个任务的失败终止整个批次 —— 失败轨迹同样是进化数据
  python3 eval/run_batch.py \
    --task "${task}" \
    --models-config "${MODELS_CONFIG}" \
    --model "${MODEL_ID}" \
    "${LOBSTER_ARGS[@]}" || echo "[run_tasks] WARN: non-zero exit for ${task}, continuing"
}

collect_new_runs() {
  # 只拷贝 marker 之后新增的 run 目录（output/ 会累积所有历史 run，全量 cp 是平方级膨胀）
  [[ -d output ]] || return 0
  while IFS= read -r d; do
    [[ -n "${d}" ]] || continue
    mkdir -p "${RESULTS_DIR}/raw/$(dirname "${d}")"
    cp -r "output/${d}" "${RESULTS_DIR}/raw/${d}"
  done < <(cd output && find . -mindepth 4 -maxdepth 4 -type d -newer "${MARKER}" | sed 's|^\./||')
  touch "${MARKER}"
}

total=0
for ((rep=1; rep<=ROLLOUTS_PER_TASK; rep++)); do
  while IFS= read -r task; do
    [[ -n "${task}" ]] || continue
    total=$((total + 1))
    echo "[run_tasks] (${total}, rep ${rep}/${ROLLOUTS_PER_TASK}) ${task}"
    if (( JOBS > 1 )); then
      run_one "${task}" &
      while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 2; done
    else
      run_one "${task}"
    fi
  done < "${LIST_FILE}"
  wait   # 并发任务全部落盘后再统一拷贝
  collect_new_runs
done

if [[ -f output/summary_all.json ]]; then
  cp output/summary_all.json "${RESULTS_DIR}/summary_all.json"
fi
echo "[run_tasks] done: ${total} task(s) -> ${RESULTS_DIR}"
