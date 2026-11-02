#!/usr/bin/env bash
# CPU-only local harness env. No model download.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q "transformers==4.44.2" safetensors numpy pytest
mkdir -p ref/hf
for f in modeling_deepseek.py configuration_deepseek.py; do
  [ -f ref/hf/$f ] || curl -sSfL "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/raw/main/$f" -o ref/hf/$f
done
echo "ok: .venv/bin/pytest -q"
