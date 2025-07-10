#!/usr/bin/env bash
# ------------------------------------------------------------------
# Video → Whisper 字幕 → Llama-3 中文摘要
# 附带 GPU 监控 (nvidia-smi dmon) + 资源统计 (/usr/bin/time -v)
# ------------------------------------------------------------------
set -eo pipefail       # 不用 -u，防止 MKL 环境变量触发 unbound

###################### 0. 解析入参 #################################
VIDEO_IN=${1:-data/samples/video_01_h264_speech.mp4}
MODEL=${2:-models/Llama/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf}

# 把路径转成绝对，避免后续 cwd 变化
VIDEO_ABS=$(realpath "$VIDEO_IN")

[[ -f $VIDEO_ABS ]] || { echo "[ERR] 视频文件不存在: $VIDEO_ABS"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/$STAMP"
mkdir -p "$LOGDIR"

###################### 1. 激活 Conda ################################
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

###################### 2. 启动 GPU 监控 #############################
nvidia-smi dmon -s pucvmet -o TD -f "$LOGDIR/gpu_dmon.csv" &
DMON_PID=$!



###################### 3. Whisper 转写 ###############################
echo "[Whisper] transcribing $VIDEO_ABS ..."
/usr/bin/time -v \
  python "$PROJECT_ROOT/src/caption/whisper_run.py" \
    --input "$VIDEO_ABS" \
  2>&1 | tee "$LOGDIR/whisper.log"

SRT="${VIDEO_ABS%.*}.srt"
[[ -f $SRT ]] || { echo "[ERR] Whisper 未生成字幕: $SRT"; kill "$DMON_PID"; exit 1; }

###################### 4. Llama-3 摘要 ##############################
echo "[Llama-3] summarising $SRT ..."
/usr/bin/time -v \
  python "$PROJECT_ROOT/src/summarizer/stream_notes.py" \
    --srt "$SRT" \
    --model "$MODEL" \
    --max-tokens 128 \
  2>&1 | tee "$LOGDIR/summarise.log"

echo "[DONE] results written to ${SRT%.srt}.summary.txt"

###################### 5. 收尾 ######################################
kill "$DMON_PID"
echo "全部日志保存在: $LOGDIR"
