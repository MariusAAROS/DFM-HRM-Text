#!/usr/bin/env bash
# Run a command in its own process group and SIGKILL the whole group if it outgrows a host-RAM budget, so a runaway
# job dies alone instead of tripping the container's out-of-memory kill (cgroup limit 128 GB, no swap: an OOM restarts
# the container and every job in it, as on 2026-11-02).
#   bash scripts/graphs/mem_guard.sh <cap_GB> -- <cmd...>
#   CONTAINER_CAP_GB=110 (default): also kill the group if the whole container passes this.
#   MEM_GUARD_KILL=0: log only (sidecar watch for GPU jobs).
# Prints the group's peak RSS on exit; exits with the command's status (137 if the guard killed it).
set -uo pipefail
cap_gb=$1; shift
[ "${1:-}" = "--" ] && shift
container_cap_gb=${CONTAINER_CAP_GB:-110}
kill_on=${MEM_GUARD_KILL:-1}
cgroup=/sys/fs/cgroup/memory.current

setsid "$@" &
pid=$!
peak_kb=0
killed=0
while kill -0 "$pid" 2>/dev/null; do
  rss_kb=$(ps -o rss= --sid "$pid" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "$rss_kb" -gt "$peak_kb" ] && peak_kb=$rss_kb
  cg_gb=$(( $(cat "$cgroup" 2>/dev/null || echo 0) / 1000000000 ))
  if [ "$rss_kb" -gt $(( cap_gb * 1000000 )) ] || [ "$cg_gb" -ge "$container_cap_gb" ]; then
    echo "[mem_guard] $(date +%T) group RSS $((rss_kb / 1000000)) GB (cap $cap_gb), container $cg_gb GB (cap $container_cap_gb)" >&2
    if [ "$kill_on" = 1 ] && [ "$killed" = 0 ]; then
      echo "[mem_guard] killing process group $pid" >&2
      kill -KILL -- "-$pid" 2>/dev/null
      killed=1
    fi
  fi
  sleep 2
done
wait "$pid"
status=$?
echo "[mem_guard] peak group RSS $((peak_kb / 1000)) MB, exit $status" >&2
exit "$status"
