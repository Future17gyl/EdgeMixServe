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
if [[ ! -f $VIDEO_IN ]]; then
  echo "[WARN] 视频文件不存在: $VIDEO_ABS"
  echo "[INFO] 正在尝试下载默认视频: video_01_h264_speech.mp4"
  mkdir -p data/samples
  yt-dlp \
    --cookies "$HOME/workspace/tools/youtube.com_cookies.txt" \
    -f "bv*[vcodec^=avc]+ba[ext=m4a]/b[ext=mp4][vcodec^=avc]" \
    -o "$HOME/workspace/EdgeMixServe/data/samples/video_01_h264_speech.mp4" \
    "https://www.youtube.com/watch?v=1aA1WGON49E" || {
      echo "[ERR] 默认视频下载失败"; exit 1;
    }
  VIDEO_IN="data/samples/video_01_h264_speech.mp4"
  VIDEO_ABS=$(realpath "$VIDEO_IN")
  echo "[INFO] 默认视频下载完成: $VIDEO_ABS"
fi

VIDEO_ABS=$(realpath "$VIDEO_IN")


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

# 根据主机名选择 Conda 环境
HOSTNAME=$(hostname) # 获取当前主机名

if [[ "$HOSTNAME" == "orin" ]]; then
    CONDA_ENV_NAME="edge_orin"
else
    # 如果不是 Orin，就使用默认的 "edge" 环境
    echo "[INFO] 非 Orin 主机，使用默认 Conda 环境 'edge'。"
    CONDA_ENV_NAME="edge"
fi

conda activate "$CONDA_ENV_NAME" || {
    echo "[ERR] 激活 Conda 环境 '$CONDA_ENV_NAME' 失败！请检查环境是否存在或名称是否正确。"
    exit 1
}

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# ==================== 4. 日志 & GPU 监控 ==========================
mkdir -p "$LOGDIR"

# 检查 nvidia-smi 命令是否存在
# command -v nvidia-smi 会返回命令的路径，如果找不到则返回非零退出状态码
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] 发现 nvidia-smi，启动 GPU 监控 ..."
    # 注意：使用完整路径可以更稳健，或者如果知道它在 PATH 中，也可以直接用 nvidia-smi
    # 这里我们假设它通过 command -v 找到了，就在 PATH 里
    nvidia-smi dmon -s pucvmet -o TD -f "$LOGDIR/gpu_dmon.csv" &
    DMON_PID=$!
    # 记录 DMON_PID，以便稍后 kill
    echo "[INFO] GPU 监控 PID: $DMON_PID"
else
    echo "[WARN] 未发现 nvidia-smi，跳过 GPU 监控。"
    DMON_PID="" # 如果没有启动监控，将 DMON_PID 设为空，以便后续判断
fi

# ============================ 5. Whisper ===========================
echo "[Whisper] transcribing $VIDEO_ABS ..."
/usr/bin/time -v \
  python "$PROJECT_ROOT/src/caption/whisper_run.py" \
    --input "$VIDEO_ABS" \
    --model_dir "$WHISPER_DIR" \
  2>&1 | tee "$LOGDIR/whisper.log"

SRT=${VIDEO_ABS%.*}.srt
[[ -f $SRT ]] || { echo "[ERR] Whisper 未生成字幕"; kill "$DMON_PID"; exit 1; }

# ============================== 6. Llama ==========================
echo "[Llama-3] summarising $SRT ..."
/usr/bin/time -v \
  python "$PROJECT_ROOT/src/summarizer/stream_notes.py" \
    --srt "$SRT" \
    --model "$LLAMA_PATH" \
    --max-tokens 128 \
  2>&1 | tee "$LOGDIR/summarise.log"

echo "[DONE] results written to ${SRT%.srt}.summary.txt"

# ============================== 7. 收尾 ===========================
kill "$DMON_PID"
echo "全部日志保存在: $LOGDIR"
