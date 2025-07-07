from faster_whisper import WhisperModel
import os

# 模型选择
model = WhisperModel("small", device="cuda", compute_type="float16")

# 视频路径
video_path = os.path.join("data", "samples", "video_01_h264.mp4")

# 推理
segments, info = model.transcribe(video_path)
segments = list(segments)  # 解决 generator 被消耗的问题

print("Detected language:", info.language)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# 保存为 SRT 字幕文件
with open("data/samples/video_01_h264.srt", "w", encoding="utf-8") as f:
    for i, segment in enumerate(segments, start=1):
        f.write(f"{i}\n")
        f.write(f"{format_time(segment.start)} --> {format_time(segment.end)}\n")
        f.write(f"{segment.text.strip()}\n\n")
