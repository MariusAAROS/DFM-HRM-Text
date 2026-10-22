#!/bin/bash
# One pass-1 worker: grep prefilter (wdt: statements, English labels/descriptions) | wikidata_pass1.py.
# Usage (from wikidata.sh via split --filter): wikidata_pass1_filter.sh <out> <worker> <n_shards>
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LC_ALL=C grep -E 'prop/direct/P|(rdf-schema#label|schema\.org/description)> ".*"@en \.$' \
  | "$REPO/.venv/bin/python" -I "$REPO/scripts/kg_pretrain/wikidata_pass1.py" "$1" "$2" "$3"
