#!/usr/bin/env bash
# ------------------------------------------------------------------
# End-to-End Pipeline
#   1. Whisper  (faster-whisper-small) → .srt
#   2. Llama-3  (8B-Q5_K_M.gguf)      → 中文摘要
#   3. GPU 监控  + /usr/bin/time      → 资源统计
# ------------------------------------------------------------------
set -eo pipefail       # 不用 -u，避免 MKL interface 变量导致报错

# ========================== 0. 参数与路径 ==========================
VIDEO_IN=${1:-data/samples/video_01_h264_speech.mp4}

# Whisper 权重
WHISPER_DIR=models/Whisper/faster-whisper-small
WHISPER_FILES=( model.bin config.json tokenizer.json vocabulary.txt )
WHISPER_REPO=https://huggingface.co/guillaumekln/faster-whisper-small/resolve/main

# Llama 权重
LLAMA_DIR=models/Llama
LLAMA_FILE=Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
LLAMA_PATH=$LLAMA_DIR/Meta-Llama-3.1-8B-Instruct-GGUF/$LLAMA_FILE
LLAMA_REPO=https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/$LLAMA_FILE

# 日志目录
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/$STAMP

# 校验视频存在
VIDEO_ABS=$(realpath "$VIDEO_IN")
[[ -f $VIDEO_ABS ]] || { echo "[ERR] 视频文件不存在: $VIDEO_ABS"; exit 1; }

# ====================== 1. Whisper 权重检查/下载 ====================
need_dl=false
for f in "${WHISPER_FILES[@]}"; do
  [[ -s $WHISPER_DIR/$f ]] || { need_dl=true; break; }
done

if $need_dl; then
  echo "[INFO] Whisper 权重缺失/不完整，补全下载 ..."
  mkdir -p "$WHISPER_DIR"
  for f in "${WHISPER_FILES[@]}"; do
    [[ -s $WHISPER_DIR/$f ]] && continue
    echo "  ↳ $f"
    wget -q --show-progress -O "$WHISPER_DIR/$f" "$WHISPER_REPO/$f" \
      || { echo "[ERR] 下载 $f 失败"; exit 1; }
  done
fi

# ======================= 2. Llama 权重检查/下载 =====================
if [[ ! -s $LLAMA_PATH ]]; then
  echo "[INFO] Llama 权重缺失/为空，开始下载 ..."
  mkdir -p "$(dirname "$LLAMA_PATH")"
  wget -q --show-progress -O "$LLAMA_PATH" "$LLAMA_REPO" \
    || { echo "[ERR] Llama 权重下载失败"; exit 1; }
fi

# =========================== 3. 环境准备 ===========================
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# 日志 & GPU 监控
mkdir -p "$LOGDIR"
nvidia-smi dmon -s pucvmet -o TD -f "$LOGDIR/gpu_dmon.csv" &
DMON_PID=$!

# ============================ 4. Whisper ===========================
echo "[Whisper] transcribing $VIDEO_ABS ..."
/usr/bin/time -v \
  python "$PROJECT_ROOT/src/caption/whisper_run.py" \
    --input "$VIDEO_ABS" \
    --model_dir "$WHISPER_DIR" \
  2>&1 | tee "$LOGDIR/whisper.log"

SRT=${VIDEO_ABS%.*}.srt
[[ -f $SRT ]] || { echo "[ERR] Whisper 未生成字幕"; kill "$DMON_PID"; exit 1; }

# ============================== 5. Llama ==========================
echo "[Llama-3] summarising $SRT ..."
/usr/bin/time -v \
  python "$PROJECT_ROOT/src/summarizer/stream_notes.py" \
    --srt "$SRT" \
    --model "$LLAMA_PATH" \
    --max-tokens 128 \
  2>&1 | tee "$LOGDIR/summarise.log"

echo "[DONE] results written to ${SRT%.srt}.summary.txt"

# ============================== 6. 收尾 ===========================
kill "$DMON_PID"
echo "全部日志保存在: $LOGDIR"
