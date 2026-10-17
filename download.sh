#!/usr/bin/env bash
# Fetch a few commaVQ shards + comma's frame decoder and verify sha256 against the HF LFS ids.
# Usage: bash download.sh [train shards...]   default: 0000 0001 0002 (+ 0039 held out as test)
set -euo pipefail
cd "$(dirname "$0")"
SHARDS=("${@:-0000 0001 0002}"); SHARDS=(${SHARDS[@]} 0039)
DS=https://huggingface.co/datasets/commaai/commavq
fetch() {  # url  dest  expected_sha256
  if [[ -f "$2" ]] && echo "$3  $2" | sha256sum -c --quiet 2>/dev/null; then echo "ok (cached) $2"; return; fi
  curl -fL --http1.1 --retry 10 --retry-all-errors --retry-delay 5 -C - -o "$2" "$1"
  echo "$3  $2" | sha256sum -c --quiet && echo "ok $2" || { echo "HASH MISMATCH $2"; exit 1; }
}
oid() { curl -s "https://huggingface.co/api/$1/tree/main" | python3 -c "import json,sys;print(next(x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']=='$2'))"; }
for s in "${SHARDS[@]}"; do fetch "$DS/resolve/main/data-$s.tar.gz" "data/raw/data-$s.tar.gz" "$(oid datasets/commaai/commavq data-$s.tar.gz)"; done
fetch "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder.onnx" data/models/decoder.onnx "$(oid models/commaai/commavq-gpt2m decoder.onnx)"
