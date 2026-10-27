"""Original-style 150-Hz gaze visualization using the public Dataset API.

Adapted from the original dataset visualization example. Retains its
per-sample rendering, participant colors, fullscreen coordinate mapping, and
last-frame extension. Local paths replace the original YAML/DataHandler setup.
Run from the repository root: python -m examples.example_visualization_code --help
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from humaneyecue.data import Dataset


# OpenCV colors are BGR; one color per participant, cycling after ten.
COLORS = [(0, 225, 0), (25, 255, 255), (0, 0, 128), (128, 0, 0),
          (230, 50, 240), (48, 130, 245), (0, 128, 128),
          (240, 240, 70), (180, 30, 145), (40, 110, 170)]


def render_visualization(dataset, video_no, question_no, video_path, output_path):
    """Write one frame per 150-Hz gaze sample, as in the source example.

    Gaze coordinates are normalized to the fullscreen display. Samples must
    retain their original positions, including invalid samples. Alignment uses
    sample indices, not timestamps; the original assumes 25-Hz source video.
    """
    gaze_by_user = dataset.gaze(video_no, question_no)
    if not gaze_by_user:
        raise ValueError('No video-segment gaze for this video/question pair')
    source = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not source.isOpened():
            raise ValueError(f'Cannot open video: {video_path}')
        fps = source.get(cv2.CAP_PROP_FPS)
        width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError('Invalid video metadata')
        # The source repeats each 25-Hz frame six times for 150-Hz gaze.
        if not math.isclose(fps, 25, abs_tol=0.01):
            raise ValueError('This source-style example requires a 25-Hz video; use humaneyecue.visualize for other frame rates')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                 150, (width, height))
        if not writer.isOpened():
            raise ValueError(f'Cannot create output: {output_path}')

        def draw_sample(frame, sample_index):
            result = frame.copy()
            for user_index, samples in enumerate(gaze_by_user.values()):
                if sample_index >= len(samples):
                    continue
                x, y = samples[sample_index]
                # Preserve sample positions but skip invalid/off-display points.
                if np.isfinite([x, y]).all() and 0 <= x < 1 and 0 <= y < 1:
                    point = (min(width - 1, math.ceil(x * width)),
                             min(height - 1, math.ceil(y * height)))
                    cv2.circle(result, point, 5, COLORS[user_index % len(COLORS)], 3)
            writer.write(result)

        sample_index = 0
        last_frame = None
        while True:
            ok, frame = source.read()
            if not ok:
                break
            last_frame = frame
            for _ in range(6):
                draw_sample(frame, sample_index)
                sample_index += 1
        if last_frame is None:
            raise ValueError('Video decoded zero frames')
        # Original example keeps remaining gaze samples on the final video frame.
        longest = max(len(samples) for samples in gaze_by_user.values())
        while sample_index < longest:
            draw_sample(last_frame, sample_index)
            sample_index += 1
        return sample_index
    finally:
        source.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--video-no', required=True, type=int)
    parser.add_argument('--question-no', required=True, type=int)
    parser.add_argument('--video', required=True, help='Matching original CLEVRER video')
    parser.add_argument('--output', default='output/example_visualization.mp4')
    args = parser.parse_args()
    frames = render_visualization(Dataset(args.data_root), args.video_no,
                                  args.question_no, args.video, args.output)
    print(f'Saved {frames} frames at 150 Hz to {args.output}')


if __name__ == '__main__':
    main()
