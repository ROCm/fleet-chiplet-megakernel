#!/usr/bin/env bash
# Download the full deepseek-ai/DeepSeek-Coder-V2-Lite-Base checkpoint (4 safetensors shards, ~31.4 GB) into
# models/DeepSeek-Coder-V2-Lite-Base/ and verify every tensor in model.safetensors.index.json is present with
# the byte count the index promises. Resumable: re-running only fetches what is missing.
# Usage: fetch_model.sh            (HF_TOKEN optional: the repo is public; set it to avoid rate limits)
# Needs huggingface_hub + safetensors in $PYTHON (default: ./.venv/bin/python from setup_local.sh, else python3).
set -euo pipefail
cd "$(dirname "$0")"
REPO=deepseek-ai/DeepSeek-Coder-V2-Lite-Base
DEST="${MODEL_DIR:-models/DeepSeek-Coder-V2-Lite-Base}"
PY=${PYTHON:-.venv/bin/python}; [ -x "$PY" ] || PY=python3
mkdir -p "$DEST"

need=$((32 * 2**30))   # 31.4 GB of shards + slack
free=$(df --output=avail -B1 "$DEST" | tail -1)
if [ ! -f "$DEST/model-00004-of-000004.safetensors" ] && [ "$free" -lt "$need" ]; then
  echo "refusing: need ~$((need / 2**30)) GiB free under $DEST, have $((free / 2**30)) GiB (models/ may be a symlink to a bigger disk)" >&2; exit 1
fi

# snapshot_download resumes partial files and skips complete ones. hf_transfer (Rust downloader) if installed.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
$PY - "$REPO" "$DEST" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1:3]
try:
    p = snapshot_download(repo, local_dir=dest, token=os.environ.get("HF_TOKEN"), max_workers=8)
except Exception as e:  # hf_transfer missing/misbehaving -> plain downloader
    if "hf_transfer" not in str(e): raise
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    p = snapshot_download(repo, local_dir=dest, token=os.environ.get("HF_TOKEN"), max_workers=8)
print("downloaded to", p)
PY

# Verify: parse each shard's safetensors header (8-byte length + JSON), check every key in the index's
# weight_map lives in the shard the index names, and the summed tensor bytes == metadata.total_size.
$PY - "$DEST" <<'PY'
import json, os, struct, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
wm, total = idx["weight_map"], idx["metadata"]["total_size"]
seen, nbytes, disk = {}, 0, 0
for shard in sorted(set(wm.values())):
    p = os.path.join(d, shard)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    disk += os.path.getsize(p)
    for k, v in hdr.items():
        if k == "__metadata__": continue
        seen[k] = shard; nbytes += v["data_offsets"][1] - v["data_offsets"][0]
    assert os.path.getsize(p) >= 8 + n + max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), f"{shard} truncated"
missing = [k for k in wm if seen.get(k) != wm[k]]
assert not missing, f"{len(missing)} tensors missing/misplaced, e.g. {missing[:3]}"
assert nbytes == total, f"tensor bytes {nbytes} != index total_size {total}"
print(f"ok: {len(wm)} tensors, {nbytes:,} tensor bytes ({nbytes/2**30:.2f} GiB), {disk:,} bytes on disk in {len(set(wm.values()))} shards")
PY
ls -la "$DEST"
