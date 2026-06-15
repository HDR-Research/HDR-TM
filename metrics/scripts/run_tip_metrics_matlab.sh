#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/mnt/data_disk/ZQL/TIP_TMO_result}"
HDR_DIR="${HDR_DIR:-/mnt/data_disk/HDRData/1000hdr/HDRtest}"
MATLAB_CODE_ROOT="${MATLAB_CODE_ROOT:-/mnt/data_disk/ZQL/MATLAB_Code}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/metric_matlab_native_last400}"
MATLAB_BIN="${MATLAB_BIN:-/usr/local/bin/matlab}"
LAST_COUNT="${LAST_COUNT:-400}"
METRIC_RESIZE_SIZE="${METRIC_RESIZE_SIZE:-512}"
RESUME="${RESUME:-1}"
KEEP_GOING="${KEEP_GOING:-1}"
METRIC_GROUPS=(${METRIC_GROUPS:-dnn tradition})

prepare_matlab_license_environment() {
  local user_id
  user_id="$(id -u)"
  if [[ -S "/run/user/${user_id}/bus" ]]; then
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${user_id}}"
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/${user_id}/bus}"
  fi
  if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X1 ]]; then
    export DISPLAY=:1
  fi
}

run_method() {
  local group="$1"
  local method="$2"
  local ldr_dir="${ROOT}/${group}/${method}"
  local report_method="${group}__${method}"
  local output_dir="${RESULT_ROOT}/1000HDR-last400/${report_method}"
  local log_file

  if [[ "${RESUME}" == "1" && -s "${output_dir}/metrics.csv" && -s "${output_dir}/summary.txt" ]]; then
    echo "[*] Reuse ${group}/${method}"
    return 0
  fi

  echo "============================================================"
  echo "[*] ${group}/${method}"
  mkdir -p "${output_dir}"
  log_file="${output_dir}/metrics_$(date '+%Y%m%d_%H%M%S').log"

  export METRICS_HDR_ROOT="${HDR_DIR}"
  export METRICS_LDR_DIR="${ldr_dir}"
  export METRICS_PRESET="hdrps_linear100_rec709"
  export METRICS_OUTPUT_CSV="${output_dir}/metrics.csv"
  export METRICS_SUMMARY_TXT="${output_dir}/summary.txt"
  export METRICS_RESIZE_SIZE="${METRIC_RESIZE_SIZE}"
  export METRICS_MAX_IMAGES="${LAST_COUNT}"
  export METRICS_MATCH_MODE="id_prefix"

  (
    cd "${MATLAB_CODE_ROOT}"
    "${MATLAB_BIN}" -batch \
      "addpath('${ROOT}'); addpath('${MATLAB_CODE_ROOT}/matlab_runner'); run_tip_database_from_env"
  ) 2>&1 | tee "${log_file}"
  return "${PIPESTATUS[0]}"
}

prepare_matlab_license_environment
mkdir -p "${RESULT_ROOT}"
overall_status=0
report_methods=()

echo "[*] MATLAB-native TMQI/TMQI-II/NLPD evaluator"
echo "[*] Reference selection: filename-sorted last ${LAST_COUNT}"
echo "[*] Result root: ${RESULT_ROOT}"

for group in "${METRIC_GROUPS[@]}"; do
  while IFS= read -r -d '' method_dir; do
    method="$(basename "${method_dir}")"
    report_methods+=("${group}__${method}")
    run_method "${group}" "${method}"
    status=$?
    if [[ "${status}" -ne 0 ]]; then
      overall_status="${status}"
      echo "[!] Failed ${group}/${method}: status=${status}"
      if [[ "${KEEP_GOING}" != "1" ]]; then
        exit "${status}"
      fi
    fi
  done < <(find "${ROOT}/${group}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done

export REPORT_RESULT_ROOT="${RESULT_ROOT}"
export REPORT_DATASETS="1000HDR-last400"
export REPORT_METHODS="${report_methods[*]}"
"${MATLAB_BIN}" -batch \
  "addpath('${MATLAB_CODE_ROOT}/matlab_runner'); write_database_reports_from_env" \
  || overall_status=$?

exit "${overall_status}"
