#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize an .srt subtitle file into one-paragraph Chinese notes
using Meta-Llama-3.1-8B-Instruct (GGUF) + llama-cpp-python.

Usage:
  python src/summarizer/stream_notes.py \
      --srt data/samples/video_01_h264.srt \
      --model models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
"""
from pathlib import Path
import argparse
from llama_cpp import Llama


def load_segments(srt_path: Path) -> str:
    """Read .srt, keep non-timestamp lines, join into long text."""
    segments = []
    with srt_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            segments.append(line)
    return " ".join(segments)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srt", required=True, help="Path to .srt subtitle file")
    parser.add_argument(
        "--model",
        default="models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        help="GGUF model path",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    srt_path = Path(args.srt).expanduser()
    text = load_segments(srt_path)

    # --- load LLM ---
    llm = Llama(
        model_path=str(Path(args.model).expanduser()),
        n_gpu_layers=-1,      # 全层 GPU
        n_ctx=4096,           # Llama-3 支持更长上下文
        chat_format="llama-3" # 使用官方模板
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个高效的会议笔记助手，请用中文输出一段简短、要点清晰的笔记摘要。",
        },
        {"role": "user", "content": text},
    ]

    resp = llm.create_chat_completion(
        messages,
        max_tokens=args.max_tokens,
        temperature=0.2,
        top_p=0.9,
    )
    summary = resp["choices"][0]["message"]["content"].strip()

    print("\n=== 摘要 ===\n", summary)

    out_path = srt_path.with_suffix(".summary.txt")
    out_path.write_text(summary, encoding="utf-8")
    print(f"\n已保存到 {out_path}")


if __name__ == "__main__":
    main()
