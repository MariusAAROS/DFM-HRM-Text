#!/bin/bash
# Resume a large download with many parallel HTTP range requests, spread over several mirrors
# (for servers that rate-limit one connection).
# Usage: fetch_ranges.sh <out_file> <n_parts> <url> [url ...]
#
# Bytes already in <out_file> are kept. The rest is split into <n_parts> parts (<out_file>.partNN, which
# may already hold a prefix from an earlier run), and each part's missing bytes are fetched in CHUNK-byte
# chunks (<out_file>.partNN.cKKKK), JOBS at a time, round-robin over the URLs. Re-running resumes.
# Verify the result against the publisher's checksum afterwards.
set -euo pipefail
out="$1" n="$2"; shift 2
urls=("$@")
CHUNK="${CHUNK:-67108864}"
JOBS="${JOBS:-24}"
total=$(curl -sIL "${urls[0]}" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
have=$(stat -c %s "$out" 2>/dev/null || echo 0)
echo "total $total have $have parts $n urls ${#urls[@]}"
[[ "$have" -lt "$total" ]] || { echo "already complete"; exit 0; }
rem=$((total - have)); step=$(( (rem + n - 1) / n ))

jobs_file="$out.jobs"
: > "$jobs_file"
k=0
for i in $(seq 0 $((n - 1))); do
  a=$((have + i * step)); b=$((a + step - 1)); [[ $b -ge $total ]] && b=$((total - 1))
  [[ $a -gt $b ]] && continue
  part="$out.part$(printf %02d "$i")"
  got=$(stat -c %s "$part" 2>/dev/null || echo 0)
  c=0
  for ((x = a + got; x <= b; x += CHUNK)); do
    y=$((x + CHUNK - 1)); [[ $y -gt $b ]] && y=$b
    echo "$x $y $part.c$(printf %04d "$c") ${urls[$((k % ${#urls[@]}))]}" >> "$jobs_file"
    c=$((c + 1)); k=$((k + 1))
  done
done
echo "$(wc -l < "$jobs_file") chunks to fetch"

fetch_chunk() {  # start end file url
  local x=$1 y=$2 f=$3 url=$4 want=$(($2 - $1 + 1)) got
  for try in $(seq 1 30); do
    got=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [[ $got -eq $want ]] && return 0
    [[ $got -gt $want ]] && rm -f "$f" && got=0
    curl -sS --fail --max-time 900 -r "$((x + got))-$y" "$url" >> "$f" || sleep 5
  done
  echo "chunk $f failed" >&2; return 1
}
export -f fetch_chunk
xargs -P "$JOBS" -L 1 bash -c 'fetch_chunk "$@"' _ < "$jobs_file"

for i in $(seq 0 $((n - 1))); do
  part="$out.part$(printf %02d "$i")"
  for c in $(ls "$part".c* 2>/dev/null | sort); do cat "$c" >> "$part" && rm "$c"; done
done
for i in $(seq 0 $((n - 1))); do
  part="$out.part$(printf %02d "$i")"
  if [[ -f "$part" ]]; then cat "$part" >> "$out" && rm "$part"; fi
done
rm -f "$jobs_file"
echo "final size $(stat -c %s "$out") expected $total"
[[ $(stat -c %s "$out") -eq $total ]]
