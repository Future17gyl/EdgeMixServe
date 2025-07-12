#!/usr/bin/env bash
# ------------------------------------------------------------------
# End-to-End: YOLO-11 (medium) 实时检测 → 逐秒聚合
#   • 自动检查/下载权重           models/YOLO/
#   • GPU 监控 (nvidia-smi dmon)  logs/<STAMP>/
#   • 资源统计 (/usr/bin/time -v)
# ------------------------------------------------------------------
set -eo pipefail

# ============================ 0. 参数 =============================
VIDEO_IN=${1:-data/samples/video_02_h264_city.mp4}
CONF=${2:-0.25}                           # 置信度阈值
YOLO_DIR=models/YOLO
YOLO_FILE=yolo11m.pt
YOLO_PATH=$YOLO_DIR/$YOLO_FILE
YOLO_REPO=https://huggingface.co/Ultralytics/YOLO11/resolve/main/$YOLO_FILE

# 校验输入视频是否存在，若不存在则自动下载默认视频
VIDEO_ABS=$(realpath "$VIDEO_IN")
if [[ ! -f $VIDEO_ABS ]]; then
  echo "[WARN] 输入视频文件不存在: $VIDEO_ABS"
  echo "[INFO] 正在下载默认示例视频: video_02_h264_city.mp4"
  mkdir -p data/samples
  yt-dlp \
    --cookies "$HOME/workspace/tools/youtube.com_cookies.txt" \
    -f "bv*[vcodec^=avc]+ba[ext=m4a]/b[ext=mp4][vcodec^=avc]" \
    -o "$HOME/workspace/EdgeMixServe/data/samples/video_02_h264_city.mp4" \
    "https://www.youtube.com/watch?v=bIo75bjq70M" || {
      echo "[ERR] 默认视频下载失败"; exit 1;
    }
  VIDEO_IN="data/samples/video_02_h264_city.mp4"
  VIDEO_ABS=$(realpath "$VIDEO_IN")
  echo "[INFO] 默认视频下载完成: $VIDEO_ABS"
fi

# ===================== 1. YOLO 权重检查/下载 ======================
if [[ ! -s $YOLO_PATH ]]; then                 # -s ⇒ 文件存在且 >0B
  echo "[INFO] YOLO 权重缺失/为空，开始下载 ..."
  mkdir -p "$YOLO_DIR"
  wget -q --show-progress -O "$YOLO_PATH" "$YOLO_REPO" \
    || { echo "[ERR] YOLO 权重下载失败"; exit 1; }
fi

# 标准化绝对路径
YOLO_PATH=$(realpath "$YOLO_PATH")

# ======================== 2. 环境准备 =============================
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# ==================== 3. 日志 & GPU 监控 ==========================
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/$STAMP; mkdir -p "$LOGDIR"
nvidia-smi dmon -s pucvmet -o TD -f "$LOGDIR/gpu_dmon.csv" &
DMON_PID=$!

# =================== 4. YOLO 实时检测 & 计时 ======================
echo "[YOLO-11M] detecting objects in $VIDEO_ABS ..."
/usr/bin/time -v \
  python "$PROJECT_ROOT/src/extractor/stream_detect.py" \
    --source "$VIDEO_ABS" \
    --model  "$YOLO_PATH" \
    --conf   "$CONF" \
  2>&1 | tee "$LOGDIR/run.log"

# ======================== 5. 收尾 ================================
kill "$DMON_PID"
echo "日志已保存至 $LOGDIR"
