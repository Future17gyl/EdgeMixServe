# EdgeMixServe

一块 Jetson/PC 边缘设备即可同时支持
• 一次性批推理（Batch / Prefill-only）
• 检索-再推理（Multi-Step / Chunked-Prefill）
• 连续流式输入（Streaming Inference）

本仓库提供最小可运行原型

1. Whisper GPU 转写（faster-whisper）
2. Meta-Llama-3 8B-Instruct ＋ TinyLLaMA-1.1B 摘要生成（llama-cpp-python）
3. demo 脚本 scripts/run\_stream\_demo.sh 串起“视频→字幕→中文笔记”全流程

## 目录结构

```
EdgeMixServe/
├── data/               示例输入与产物（大文件已忽略）
│   └── samples/        video_01_h264.mp4 / .srt / .summary.txt
├── env/                Conda 环境描述
│   └── conda/edge.yml
├── models/             本地 GGUF 权重（不提交）
│   ├── Meta-Llama-3.1-8B-Instruct-GGUF/
│   └── tinyllama/
├── src/                代码主体
│   ├── caption/whisper_run.py
│   ├── summarizer/stream_notes.py
│   └── …
├── scripts/run_stream_demo.sh
└── README.txt
```

## 快速开始

1. 创建 Conda 环境

```
conda env create -f env/conda/edge.yml
conda activate edge
```

2. 下载权重

```
# Whisper small.en-int8
python -m faster_whisper.utils.download --model small

# Llama-3-8B-Instruct (Q5_K_M)
huggingface-cli login            # 首次需输入 HF token
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --local-dir models/Meta-Llama-3.1-8B-Instruct-GGUF \
  --include Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
```

3. 安装 GPU 版 llama-cpp-python

```
pip uninstall -y llama-cpp-python llama_cpp
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --no-binary :all: llama-cpp-python
```

4. 运行 Demo

```
./scripts/run_stream_demo.sh data/samples/video_01_h264.mp4
```

完成后生成
- video_01_h264.srt – Whisper 字幕
- video_01_h264.summary.txt – Llama-3 中文笔记

## 单元脚本

| 功能        | 路径                              | 说明                 |
| ----------- | -------------------------------- | -------------------- |
| 视频转写     | src/caption/whisper\_run.py      | mp4 → .srt           |
| 字幕摘要     | src/summarizer/stream\_notes.py  | .srt → 中文摘要       |
| 全链路 Demo  | scripts/run\_stream\_demo.sh     | 串联 Whisper + LLM    |

## 性能提示

- n_gpu_layers = -1（所有模型层都加载到GPU）
- n_ctx = 4096（上下文窗口大小，即最大处理的token数）
- Q5_K_M 模型显存 ≈ 7 GB
- 监控命令：`nvidia-smi dmon -s pucvmet`

## TODO

- YOLOv8 / SigLIP 关键帧抽取
- Paged KV-Cache 合并
- SM-Slice 调度与 Goodput 指标
- Jetson Orin 功耗 / 延迟评估
