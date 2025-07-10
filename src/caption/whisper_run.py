#!/usr/bin/env python3
"""
Whisper inference helper.

例子:
    python -m caption.whisper_run --input data/samples/video_01_h264.mp4
"""

from pathlib import Path
import argparse, os, sys
from faster_whisper import WhisperModel


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入视频/音频路径")
    ap.add_argument("--model_size", default="small", choices=["tiny", "base", "small", "medium", "large"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float16", help="faster-whisper compute_type")
    args = ap.parse_args()

    video_path = Path(args.input).expanduser().resolve()
    if not video_path.is_file():
        sys.exit(f"[ERR] 找不到输入文件: {video_path}")

    srt_path = video_path.with_suffix(".srt")

    model = WhisperModel(args.model_size, device=args.device, compute_type=args.dtype)

    print(f">> Transcribing {video_path} ...")
    segments, info = model.transcribe(str(video_path))
    segments = list(segments)  # materialise generator

    print("Detected language:", info.language)
    for seg in segments:
        print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")

    # --- write SRT ------------------------------------------------
    with srt_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")

    print(f">> SRT saved to {srt_path}")


if __name__ == "__main__":
    main()
