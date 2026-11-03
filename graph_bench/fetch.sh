#!/usr/bin/env bash
# Download the raw sources of the Mimir graph benchmark (wiki: mimir-graph-benchmark) into
# data/graphs/bench/raw/<source>/, one empty directory per source, each with a SOURCE.json (urls, date, sha256).
#   bash graph_bench/fetch.sh all            # or a subset: bash graph_bench/fetch.sh webqsp cwq clutrr
# Network only (no GPU). Detach the full run: setsid nohup bash graph_bench/fetch.sh all > <log> 2>&1 < /dev/null &
# Converters (graph_bench/sources/*) run one at a time, under the memory guard (the container has 128 GB, no swap;
# an out-of-memory kill restarts it with every job, 2026-11-02): bash scripts/graphs/mem_guard.sh 24 -- .venv/bin/python -m ...
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RAW=${RAW:-$REPO/data/graphs/bench/raw}
PY=${PY:-$REPO/.venv/bin/python}

hf() {  # hf <dir> <repo> [allow_patterns...]
  local dir=$1 repo=$2; shift 2
  "$PY" -I - "$RAW/$dir" "$repo" "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download
out, repo, pats = sys.argv[1], sys.argv[2], sys.argv[3:] or None
snapshot_download(repo, repo_type="dataset", local_dir=out, allow_patterns=pats)
EOF
}

url() {  # url <dir> <url>...
  local dir=$1; shift
  for u in "$@"; do curl -fsSL --retry 3 -o "$RAW/$dir/$(basename "${u%%\?*}")" "$u"; done
}

gdrive() {  # gdrive <dir> <file id> <name>
  "$PY" -I -c "import gdown, sys; gdown.download(id=sys.argv[1], output=sys.argv[2], quiet=False)" "$2" "$RAW/$1/$3"
}

gdrive_folder() {  # gdrive_folder <dir> <folder url>
  "$PY" -I -c "import gdown, sys; gdown.download_folder(url=sys.argv[1], output=sys.argv[2], quiet=False)" "$2" "$RAW/$1"
}

provenance() {  # provenance <dir> <source description>
  "$PY" -I - "$RAW/$1" "$2" <<'EOF'
import datetime, hashlib, json, os, sys
root, desc = sys.argv[1], sys.argv[2]
files = {}
for dp, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(dp, f)
        if f == "SOURCE.json" or "/.cache/" in p:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        files[os.path.relpath(p, root)] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
json.dump({"source": desc, "fetched": datetime.date.today().isoformat(), "files": files},
          open(os.path.join(root, "SOURCE.json"), "w"), indent=1)
print(f"{root}: {len(files)} files")
EOF
}

fetch() {
  local s=$1
  if [[ -f "$RAW/$s/SOURCE.json" ]]; then echo "[skip] $s (SOURCE.json exists)"; return; fi
  mkdir -p "$RAW/$s"
  echo "[fetch] $s"
  case $s in
    webqsp) hf webqsp rmanluo/RoG-webqsp "data/*"; provenance webqsp "hf:rmanluo/RoG-webqsp" ;;
    cwq) hf cwq rmanluo/RoG-cwq "data/*"; provenance cwq "hf:rmanluo/RoG-cwq" ;;
    clutrr)
      for t in gen_train234_test2to10 gen_train23_test2to10; do
        mkdir -p "$RAW/clutrr/$t"
        for sp in train validation test; do
          curl -fsSL --retry 3 -o "$RAW/clutrr/$t/$sp.csv" \
            "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/$t/$sp.csv"
        done
      done
      provenance clutrr "github:kliang5/CLUTRR_huggingface_dataset (CLUTRR v1 csv)" ;;
    clrs) hf clrs/train smcleish/CLRS-Text-train "data/*"; hf clrs/test smcleish/CLRS-Text-test "data/*"
      provenance clrs "hf:smcleish/CLRS-Text-train, hf:smcleish/CLRS-Text-test" ;;
    graphinstruct) hf graphinstruct GraphWiz/GraphInstruct-Test "*.json"
      hf graphinstruct/train GraphWiz/GraphInstruct "*.json"
      provenance graphinstruct "hf:GraphWiz/GraphInstruct-Test, hf:GraphWiz/GraphInstruct" ;;
    nlgraph)
      git clone -q --depth 1 https://github.com/Arthur-Heng/NLGraph "$RAW/nlgraph/repo"
      rm -rf "$RAW/nlgraph/repo/.git"; provenance nlgraph "github:Arthur-Heng/NLGraph (main, depth 1)" ;;
    mintaka) url mintaka https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_{train,dev,test}.json
      provenance mintaka "github:amazon-science/mintaka data/" ;;
    pathquestions)
      url pathquestions https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/{PQ-2H,PQ-3H,PQL-2H,PQL-3H,PQL-3H_more,2H-kb,3H-kb,PQL2-KB,PQL3-KB}.txt
      provenance pathquestions "github:zmtkeke/IRN PathQuestion/" ;;
    lcquad2) url lcquad2 https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15738824/train.json \
               https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15738818/test.json
      provenance lcquad2 "figshare LC-QuAD 2.0 train/test json" ;;
    cronquestions) gdrive cronquestions 1fe7-x7ChszqzczKncoZcpwmWc1PBq1_0 data_v2.zip
      (cd "$RAW/cronquestions" && unzip -q data_v2.zip && rm data_v2.zip)
      provenance cronquestions "gdrive CronKGQA data_v2.zip (README: fixed {tail2} version)" ;;
    grapharena) gdrive_folder grapharena "https://drive.google.com/drive/folders/1mvJSUTrfOX13wgpkyb3w8s_SJqipnb1c"
      provenance grapharena "gdrive GraphArena folder (dataset.zip, source.zip, results.zip)" ;;
    *) echo "unknown source $s"; return 1 ;;
  esac
}

ALL=(webqsp cwq clutrr clrs graphinstruct nlgraph mintaka pathquestions lcquad2 cronquestions grapharena)
# Each source runs in its own shell: set -e is ignored inside a function called from `||`, which would let a failed
# download still write SOURCE.json (and be skipped next time).
if [[ ${1:-} == _one ]]; then fetch "$2"; exit; fi
[[ $# -eq 0 ]] && { echo "usage: $0 all | ${ALL[*]}"; exit 1; }
[[ $1 == all ]] && set -- "${ALL[@]}"
failed=()
for s in "$@"; do
  if ! bash "$0" _one "$s"; then echo "[fail] $s"; failed+=("$s"); fi
done
echo "done; failed: ${failed[*]:-none}"
