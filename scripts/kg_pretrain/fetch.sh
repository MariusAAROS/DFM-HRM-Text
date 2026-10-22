#!/bin/bash
# Download the Tier 1+2 KG / graph sources of corpus_graphes_llm.csv into data/kg_pretrain/raw/<source>/.
#
# Usage: bash scripts/kg_pretrain/fetch.sh <source>... | all
# Each source gets its own directory and a SOURCE.json (urls, date, sha256, license).
# A source whose SOURCE.json exists is skipped; delete it to re-fetch. Downloads resume (wget -c).
# Run detached: setsid nohup bash scripts/kg_pretrain/fetch.sh all > data/kg_pretrain/logs/fetch.log 2>&1 < /dev/null &
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="${KG_RAW:-$REPO/data/kg_pretrain/raw}"
PY="$REPO/.venv/bin/python"
WIKIDATA_DUMP="${WIKIDATA_DUMP:-20261014}"

# name|license|url[ out_name] ...
declare -A SRC_LICENSE SRC_URLS
SRC_LICENSE[conceptnet]="CC BY-SA 4.0 (ConceptNet 5.7; some sources CC BY 4.0)"
SRC_URLS[conceptnet]="https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
SRC_LICENSE[atomic2020]="CC BY 4.0"
SRC_URLS[atomic2020]="https://drive.usercontent.google.com/download?id=1uuY0Y_s8dhxdsoOe8OHRgsqf-9qJIai7&export=download&confirm=t>atomic2020_data-feb2021.zip"
SRC_LICENSE[hetionet]="CC0 1.0 (the hetnet; some integrated sources carry their own terms, see hetionet README)"
SRC_URLS[hetionet]="https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
SRC_LICENSE[primekg]="CC0 1.0 (Harvard Dataverse doi:10.7910/DVN/IXA7BM)"
SRC_URLS[primekg]="https://dataverse.harvard.edu/api/access/datafile/6180620>kg.csv https://dataverse.harvard.edu/api/access/datafile/6180617>nodes.tab https://dataverse.harvard.edu/api/access/datafile/6191270>README.txt"
SRC_LICENSE[wikidata5m]="CC BY 4.0"
SRC_URLS[wikidata5m]="https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1>wikidata5m_text.txt.gz https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1>wikidata5m_alias.tar.gz https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1>wikidata5m_transductive.tar.gz"
SRC_LICENSE[kelm]="CC BY-SA 2.0 (KELM corpus README; to verify)"
SRC_URLS[kelm]="https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/kelm_generated_corpus.jsonl"
SRC_LICENSE[trex]="CC BY 4.0 (figshare 5146864)"
SRC_URLS[trex]="https://ndownloader.figshare.com/files/8760241>T-REx.zip"
SRC_LICENSE[yago]="CC BY 4.0"
SRC_URLS[yago]="$(for f in facts beyond-wikipedia labels beyond-wikipedia-labels schema taxonomy; do printf 'https://yago-knowledge.org/data/yago4.6/yago-4.6-%s.zip ' "$f"; done)"
SRC_LICENSE[wikidata]="CC0 1.0"
SRC_URLS[wikidata]="https://dumps.wikimedia.org/wikidatawiki/entities/${WIKIDATA_DUMP}/wikidata-${WIKIDATA_DUMP}-truthy-BETA.nt.bz2"
# Hugging Face datasets: hf:<repo> (snapshot_download; only the train files, never -Test or -DPO).
SRC_LICENSE[graphinstruct]="not stated on the HF card (to verify)"
SRC_URLS[graphinstruct]="hf:GraphWiz/GraphInstruct hf:GraphWiz/GraphInstruct-RFT-72K"
# GraphQA generator code (Apache 2.0); generation itself is run by gen_graphqa.sh.
SRC_LICENSE[graphqa_gen]="Apache 2.0"
SRC_URLS[graphqa_gen]="git:https://github.com/google-research/google-research.git#graphqa"

ORDER=(conceptnet atomic2020 hetionet primekg graphinstruct graphqa_gen wikidata5m trex kelm yago wikidata)

fetch_one() {
  local name="$1" dir="$RAW/$1"
  if [[ -f "$dir/SOURCE.json" ]]; then echo "[$name] already fetched, skipping"; return; fi
  mkdir -p "$dir"
  local spec url out
  for spec in ${SRC_URLS[$name]}; do
    case "$spec" in
      hf:*)
        HF_HUB_DISABLE_TELEMETRY=1 "$PY" -I -c '
import sys
from huggingface_hub import snapshot_download
repo, out = sys.argv[1], sys.argv[2]
p = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out)
print(p)' "${spec#hf:}" "$dir/${spec#hf:GraphWiz/}"
        ;;
      git:*)
        url="${spec#git:}"
        if [[ ! -d "$dir/google-research" ]]; then
          git clone --depth 1 --filter=blob:none --sparse "${url%%#*}" "$dir/google-research"
          git -C "$dir/google-research" sparse-checkout set "${url##*#}"
        fi
        ;;
      *)
        url="${spec%%>*}"
        if [[ "$spec" == *">"* ]]; then out="${spec##*>}"; else out="$(basename "${url%%\?*}")"; fi
        echo "[$name] $url -> $out"
        wget -c --no-verbose --tries=20 --retry-connrefused --waitretry=30 -O "$dir/$out" "$url"
        ;;
    esac
  done
  "$PY" -I - "$dir" "$name" "${SRC_LICENSE[$name]}" "${SRC_URLS[$name]}" <<'EOF'
import datetime, hashlib, json, os, sys
d, name, lic, urls = sys.argv[1:5]
files = {}
for root, dirs, fs in os.walk(d):
    dirs[:] = [x for x in dirs if x not in (".git", ".cache")]
    for f in sorted(fs):
        p = os.path.join(root, f)
        if f == "SOURCE.json" or os.path.getsize(p) == 0:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        files[os.path.relpath(p, d)] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
json.dump({"source": name, "license": lic, "urls": urls.split(),
           "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "files": files}, open(os.path.join(d, "SOURCE.json"), "w"), indent=1)
print(f"[{name}] {len(files)} files, {sum(v['bytes'] for v in files.values()) / 1e9:.2f} GB")
EOF
}

targets=("$@")
[[ ${#targets[@]} -eq 0 || "${targets[0]}" == all ]] && targets=("${ORDER[@]}")
for t in "${targets[@]}"; do
  [[ -n "${SRC_URLS[$t]:-}" ]] || { echo "unknown source: $t (known: ${ORDER[*]})" >&2; exit 1; }
  fetch_one "$t"
done
echo "fetch done: ${targets[*]}"
