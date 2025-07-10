#!/usr/bin/env python3
"""
Real-time object detection, 1-second sliding window aggregation.
Usage: python -m extractor.stream_detect --source path/to/video.mp4
"""
import argparse, time, logging, pathlib, json, cv2
from collections import Counter, defaultdict
from ultralytics import YOLO      # pip install ultralytics
from datetime import datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='video file or rtsp url')
    ap.add_argument('--model', default='yolov12n.pt')   # or yolov9n.pt
    ap.add_argument('--conf',  type=float, default=0.25)
    args = ap.parse_args()

    # -------- logging ----------
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = pathlib.Path('logs') / stamp
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / 'stream.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()]
    )
    logging.info(f'# Start {args.source} with {args.model}')

    # -------- video & model ----
    cap   = cv2.VideoCapture(args.source)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    model = YOLO(args.model)

    sec_bucket = defaultdict(Counter)   # {sec_idx: Counter({class:count})}
    frame_idx  = 0
    t0         = time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break

        # 推理（一次 1 帧；也可 batch 推）
        results = model.predict(
            source=frame, stream=False, conf=args.conf, verbose=False
        )[0]
        cls_names = [results.names[int(c)] for c in results.boxes.cls]

        sec = int(frame_idx / fps)
        sec_bucket[sec].update(cls_names)

        # 每秒结束后 flush
        if frame_idx % int(fps) == int(fps) - 1:
            summary = dict(sec=sec,
                           time=f'{sec:04d}s',
                           objs=sec_bucket[sec])
            logging.info(json.dumps(summary, ensure_ascii=False))
            del sec_bucket[sec]      # 释放内存

        frame_idx += 1

    cap.release()
    logging.info(f'# Done in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
