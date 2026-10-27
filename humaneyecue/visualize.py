"""Overlay public gaze samples on a supplied CLEVRER video."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from .data import Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--video-no', type=int, required=True)
    parser.add_argument('--question-no', type=int, required=True)
    parser.add_argument('--video', required=True, help='Actual source video path; no assumption about CLEVRER vs selected IDs')
    parser.add_argument('--output', default='output/gaze_overlay.mp4')
    args = parser.parse_args()
    users = Dataset(args.data_root).gaze(args.video_no, args.question_no)
    if not users:
        raise ValueError('No video-segment gaze data for this pair')
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {args.video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = (int(capture.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError('Invalid video metadata')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise ValueError('Cannot create MP4 output')
    palette = [(52, 180, 245), (194, 118, 46), (106, 190, 91), (180, 86, 160)]
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            first = round(frame_idx * 150 / fps)
            last = round((frame_idx + 1) * 150 / fps)
            for i, samples in enumerate(users.values()):
                for x, y in samples[first:last]:
                    if np.isfinite([x, y]).all() and 0 <= x < 1 and 0 <= y < 1:
                        cv2.circle(frame, (int(x * width), int(y * height)), 3, palette[i % len(palette)], -1)
            writer.write(frame)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()
    if not frame_idx:
        raise ValueError('Video decoded zero frames')
    print(f'Saved {frame_idx} frames to {args.output}')


if __name__ == '__main__':
    main()
