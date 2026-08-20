#!/usr/bin/env bash
# 窗口 1：模型服务。用法:
#   bash w1_server.sh [base|rl] [0p6b|4b|8b]     # 默认 base 8b
#   bash w1_server.sh down                        # 停服务
# sglang 后台运行，日志在 results/logs/，本窗口随即空闲可查日志:
#   tail -f $ABLATION_ROOT/results/logs/sglang_*.log
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/env.sh"

ROLE="${1:-base}"
SIZE="${2:-${SIZE}}"

exec bash "${ABLATION_ROOT}/scripts/up.sh" "${ROLE}" "${SIZE}"
