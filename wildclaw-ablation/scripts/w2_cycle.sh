#!/usr/bin/env bash
# 窗口 2：跑消融闭环（采集 -> 进化 -> 评测 -> 汇总）。前置: 窗口 1 模型服务已就绪。
# 用法:
#   bash w2_cycle.sh                 # 默认 SIZE=8b MODE=skill_only ROLLOUTS=4，前台跑
#   SIZE=4b bash w2_cycle.sh         # 换模型尺寸
#   MODE=full DO_TRAIN=1 bash w2_cycle.sh   # 完整四组消融 + 离线 GRPO
#   BG=1 bash w2_cycle.sh            # 后台跑（nohup + 日志），窗口解放
# 想先小样本验证进化修复:
#   ROLLOUTS_PER_TASK=1 WCB_CATEGORIES="01_Productivity_Flow" bash w2_cycle.sh
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/env.sh"

# benchmark 的 python 环境
source "${WCB_ROOT}/.venv/bin/activate"
cd "${WCB_ROOT}"

LOG_DIR="${ABLATION_ROOT}/results/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/cycle_${SIZE}_${MODE}_$(date +%Y%m%d_%H%M).log"

if [[ "${BG:-0}" == "1" ]]; then
  echo "[w2] background -> ${LOG_FILE}"
  nohup bash "${ABLATION_ROOT}/scripts/run_cycle.sh" > "${LOG_FILE}" 2>&1 &
  echo "[w2] pid=$! ; tail -f ${LOG_FILE}"
else
  echo "[w2] foreground, log -> ${LOG_FILE}"
  bash "${ABLATION_ROOT}/scripts/run_cycle.sh" 2>&1 | tee "${LOG_FILE}"
fi
