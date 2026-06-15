#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/metric_results_last400}"
LOG_ROOT="${RESULT_ROOT}/runner_logs"
mkdir -p "${LOG_ROOT}"

echo "[*] Starting DNN metrics on GPU 0"
METRIC_GROUPS="dnn" \
GENERATE_REPORT=0 \
CUDA_VISIBLE_DEVICES=0 \
RESULT_ROOT="${RESULT_ROOT}" \
bash "${ROOT}/run_tip_metrics.sh" > "${LOG_ROOT}/dnn.log" 2>&1 &
dnn_pid=$!

echo "[*] Starting traditional metrics on GPU 1"
METRIC_GROUPS="tradition" \
GENERATE_REPORT=0 \
CUDA_VISIBLE_DEVICES=1 \
RESULT_ROOT="${RESULT_ROOT}" \
bash "${ROOT}/run_tip_metrics.sh" > "${LOG_ROOT}/tradition.log" 2>&1 &
tradition_pid=$!

dnn_status=0
tradition_status=0
wait "${dnn_pid}" || dnn_status=$?
wait "${tradition_pid}" || tradition_status=$?

source /home/yan/miniconda3/etc/profile.d/conda.sh
conda activate hdrtmo
python "${ROOT}/compute_tip_metrics.py" report --result_root "${RESULT_ROOT}"
report_status=$?

echo "[*] DNN status: ${dnn_status}"
echo "[*] Traditional status: ${tradition_status}"
echo "[*] Report status: ${report_status}"

if [[ "${dnn_status}" -ne 0 ]]; then
  exit "${dnn_status}"
fi
if [[ "${tradition_status}" -ne 0 ]]; then
  exit "${tradition_status}"
fi
exit "${report_status}"
