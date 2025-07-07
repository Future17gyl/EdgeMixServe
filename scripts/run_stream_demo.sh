#!/usr/bin/env bash
# scripts/run_stream_demo.sh
# ----------------------------------------------------------
# Quick demo: transcribe sample video with Whisper and
# summarise it into bullet notes with TinyLlama (GGUF).
# ----------------------------------------------------------

set -e                     # 出错立即退出
SAMPLE="data/samples/sample20s.mp4"

# 1) 激活 conda 环境
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

# 2) 跑最小流水线
python src/summarizer/stream_notes.py "$SAMPLE"
