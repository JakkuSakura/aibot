#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="${SCRIPT_DIR}/main.py"
UV_BIN="${UV_BIN:-uv}"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config.toml}"

if [[ ! -f "${MAIN_PY}" ]]; then
  echo "[watchdog] main.py not found at ${MAIN_PY}" >&2
  exit 1
fi

child_pid=""

start_main() {
  echo "[watchdog] starting main.py with config=${CONFIG_FILE}"
  (
    cd "${SCRIPT_DIR}" || exit 1
    "${UV_BIN}" run aibot --config "${CONFIG_FILE}" "$@"
  ) &
  child_pid=$!
}

stop_child() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    echo "[watchdog] stopping main.py (pid=${child_pid})"
    kill "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
}

trap 'stop_child; exit 0' INT TERM
trap 'stop_child' EXIT

while true; do
  start_main "$@"
  wait "${child_pid}"
  exit_code=$?
  echo "[watchdog] main.py exited with code ${exit_code}; restarting in 1s"
  sleep 1
done
