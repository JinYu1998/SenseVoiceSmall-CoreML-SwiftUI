#!/usr/bin/env bash

set -euo pipefail

# 获取脚本所在目录
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 模型目录（在脚本目录下）
MODEL_ROOT_DIR="$SCRIPT_DIR/Model"
MODEL_BUNDLE_DIR="$MODEL_ROOT_DIR/SenseVoiceSmall300.mlmodelc"

REPO_ID="Mor1998/SenseVoiceSmall300"

# 检查 hf CLI
if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found."
  echo "Install it first with:"
  echo "  pip install -U \"huggingface_hub[cli]\""
  exit 1
fi

mkdir -p "$MODEL_ROOT_DIR"

echo "Downloading model from Hugging Face: $REPO_ID"
echo "Target directory: $MODEL_ROOT_DIR"

hf download "$REPO_ID" --local-dir "$MODEL_ROOT_DIR"

# 如果仓库是散文件结构，整理成 .mlmodelc bundle
if [ ! -d "$MODEL_BUNDLE_DIR" ] && [ -f "$MODEL_ROOT_DIR/model.mil" ]; then
  echo "Reorganizing files into mlmodelc bundle..."

  mkdir -p "$MODEL_BUNDLE_DIR"

  for item in analytics coremldata.bin metadata.json model.mil README.md weights; do
    if [ -e "$MODEL_ROOT_DIR/$item" ]; then
      mv "$MODEL_ROOT_DIR/$item" "$MODEL_BUNDLE_DIR/"
    fi
  done
fi

# 最终检查
if [ ! -d "$MODEL_BUNDLE_DIR" ]; then
  echo "❌ Error: $MODEL_BUNDLE_DIR not found after download."
  echo "Please check the Hugging Face repository structure."
  exit 1
fi

echo "✅ Model download complete."
echo "📁 Location: $MODEL_BUNDLE_DIR"