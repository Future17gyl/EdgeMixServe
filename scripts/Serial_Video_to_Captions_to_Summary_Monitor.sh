#!/usr/bin/env bash
# -------------------------------------------------------------
# 监控版流水线：
#   1) 启动 nvidia-smi dmon → 每秒记录 GPU 使用
#   2) /usr/bin/time -v 运行主脚本
#   3) 收尾：杀监控、汇总日志
# -------------------------------------------------------------
set -e

## -------- 0. 日志目录 & 入参 ---------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/$STAMP"
mkdir -p "$LOGDIR"

VIDEO=${1:-data/samples/video_01_h264.mp4}

## -------- 1. 激活环境 ----------------------------------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

## -------- 2. GPU 监控后台 ------------------------------------
# P:功耗  U:GPU利用  C:核心时钟  V:显存时钟  M:显存利用  E:显存占用  T:温度
nvidia-smi dmon -s pucvmet -o TD -f "$LOGDIR/gpu_dmon.csv" &
DMON_PID=$!

## -------- 3. 运行主流水线并统计 CPU/内存 ----------------------
/usr/bin/time -v bash scripts/Serial_Video_to_Captions_to_Summary.sh "$VIDEO" \
  2>&1 | tee "$LOGDIR/run.log"

## -------- 4. 清理 & 后处理 -----------------------------------
kill "$DMON_PID"

echo "日志已保存至 $LOGDIR"
