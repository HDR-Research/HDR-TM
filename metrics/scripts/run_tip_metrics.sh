#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HDR_DIR="${HDR_DIR:-/mnt/data_disk/HDRData/1000hdr/HDRtest}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/metric_results_last400}"
LAST_COUNT="${LAST_COUNT:-400}"
METRIC_RESIZE_SIZE="${METRIC_RESIZE_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RESUME="${RESUME:-1}"
KEEP_GOING="${KEEP_GOING:-1}"
GENERATE_REPORT="${GENERATE_REPORT:-1}"
METRIC_GROUPS=(${METRIC_GROUPS:-dnn tradition})

source /home/yan/miniconda3/etc/profile.d/conda.sh
conda activate hdrtmo
export CUDA_VISIBLE_DEVICES

mkdir -p "${RESULT_ROOT}"
overall_status=0

echo "[*] Conda environment: hdrtmo"
echo "[*] HDR references: ${HDR_DIR}"
echo "[*] Selection: filename-sorted last ${LAST_COUNT}"
echo "[*] Results: ${RESULT_ROOT}"
echo "[*] Device: ${DEVICE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for group in "${METRIC_GROUPS[@]}"; do
  group_dir="${ROOT}/${group}"
  [[ -d "${group_dir}" ]] || continue

  while IFS= read -r -d '' method_dir; do
    method="$(basename "${method_dir}")"
    output_dir="${RESULT_ROOT}/${group}/${method}"
    log_file="${output_dir}/run.log"

    if [[ "${RESUME}" == "1" && -s "${output_dir}/metrics.csv" && -s "${output_dir}/summary.json" ]]; then
      echo "[*] Reuse ${group}/${method}"
      continue
    fi

    echo "============================================================"
    echo "[*] ${group}/${method}"
    mkdir -p "${output_dir}"
    python -u "${ROOT}/compute_tip_metrics.py" compute \
      --hdr_dir "${HDR_DIR}" \
      --ldr_dir "${method_dir}" \
      --output_dir "${output_dir}" \
      --group "${group}" \
      --method "${method}" \
      --last_count "${LAST_COUNT}" \
      --metric_resize_size "${METRIC_RESIZE_SIZE}" \
      --device "${DEVICE}" \
      2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    if [[ "${status}" -ne 0 ]]; then
      overall_status="${status}"
      echo "[!] Failed ${group}/${method}: ${status}"
      if [[ "${KEEP_GOING}" != "1" ]]; then
        exit "${status}"
      fi
    fi
  done < <(find "${group_dir}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done

if [[ "${GENERATE_REPORT}" == "1" ]]; then
  python "${ROOT}/compute_tip_metrics.py" report --result_root "${RESULT_ROOT}"
  report_status=$?
  if [[ "${report_status}" -ne 0 ]]; then
    overall_status="${report_status}"
  fi
fi

exit "${overall_status}"
