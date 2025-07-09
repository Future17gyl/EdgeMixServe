#!/usr/bin/env bash
# scripts/Serial_Video_to_Captions_to_Summary.sh
# ----------------------------------------------------------
# 输入 MP4 → Whisper 转写 → Llama-3 中文摘要
# 生成：
#   *.srt            字幕文件
#   *.summary.txt    一段中文笔记
# ----------------------------------------------------------

set -e  # 出错立即退出

# --------- 0. 解析入参 ------------------------------------
VIDEO=${1:-data/samples/video_01_h264.mp4}        # 未传参则用默认样例
MODEL=models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf

# --------- 1. 激活 conda 环境 ------------------------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

# --------- 2. Whisper 生成字幕 -----------------------------
echo "[Whisper] transcribing $VIDEO ..."
python src/caption/whisper_run.py -i "$VIDEO"

# 得到与视频同名的 .srt
SRT="${VIDEO%.*}.srt"

# --------- 3. Llama-3 生成摘要 -----------------------------
echo "[Llama-3] summarising $SRT ..."
python src/summarizer/stream_notes.py \
  --srt "$SRT" \
  --model "$MODEL" \
  --max-tokens 128

echo "[DONE] results written to ${SRT%.srt}.summary.txt"
