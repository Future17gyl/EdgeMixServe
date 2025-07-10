#!/usr/bin/env bash
# ------------------------------------------------------------------
# Video → YOLO-11 实时检测（逐秒聚合）
# 权重统一存放在 models/YOLO/ 下；若 <1 MB 则视为残缺并重新下载
# ------------------------------------------------------------------
set -eo pipefail

###################### 0. 解析入参 #################################
VIDEO_IN=${1:-data/samples/video_02_h264_city.mp4}
CONF=${2:-0.25}                           # 置信度阈值
MODEL_DIR=models/YOLO
MODEL=${3:-$MODEL_DIR/yolo11m.pt}         # 默认 YOLO-11-M

VIDEO_ABS=$(realpath "$VIDEO_IN")
[[ -f $VIDEO_ABS ]] || { echo "[ERR] 视频文件不存在: $VIDEO_ABS"; exit 1; }

################ 权重检查 & 自动下载 (≥1 MB 视为完整) ###############
MIN_SIZE=1048576  # 1 MB

download_weight () {
  mkdir -p "$MODEL_DIR"
  echo "[INFO] 正在下载 $MODEL ..."
  wget -q --show-progress -O "$MODEL" \
       https://huggingface.co/Ultralytics/YOLO11/resolve/main/$(basename "$MODEL")
}

need_redownload=true
if [[ -f $MODEL ]]; then
  size=$(stat -c%s "$MODEL" 2>/dev/null || echo 0)
  if (( size >= MIN_SIZE )); then
    need_redownload=false
  else
    echo "[WARN] 检测到残缺权重文件（大小 $size 字节），重新下载"
    rm -f "$MODEL"
  fi
fi
$need_redownload && download_weight || true
MODEL_PATH=$(realpath "$MODEL")

###################### 1. 激活 Conda ################################
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

###################### 2. 日志目录 & GPU 监控 #######################
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/$STAMP"; mkdir -p "$LOGDIR"

nvidia-smi dmon -s pucvmet -o TD -f "$LOGDIR/gpu_dmon.csv" &
DMON_PID=$!

###################### 3. 实时检测 & 资源统计 #######################
echo "[YOLO-11M] detecting objects in $VIDEO_ABS ..."
/usr/bin/time -v \
  python "$PROJECT_ROOT/src/extractor/stream_detect.py" \
    --source "$VIDEO_ABS" \
    --model  "$MODEL_PATH" \
    --conf   "$CONF" \
  2>&1 | tee "$LOGDIR/run.log"

###################### 4. 收尾 ######################################
kill "$DMON_PID"
echo "日志已保存至 $LOGDIR"
