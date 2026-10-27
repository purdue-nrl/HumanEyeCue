"""Object-level gaze assignment using the source CLEVRER mask decoder."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from .data import Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--clevrer-root', required=True)
    parser.add_argument('--output', default='output/objects.npz')
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / 'analysis' / 'handle_CLEVRER.py'
    spec = importlib.util.spec_from_file_location('clevrer_masks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clevrer = module.CLEVRER(root=Path(args.clevrer_root).as_posix().rstrip('/') + '/')
    dataset = Dataset(args.data_root)
    output = {}
    for video, pairs in dataset.questions().groupby('video_no'):
        masks = clevrer.get_all_per_frame_masks(video_idx=int(video), grow_mask_by_n_pixels=30)
        for question in pairs.question_no:
            for uid, gaze in dataset.gaze(int(video), int(question)).items():
                hits = np.full((len(gaze), masks.shape[1]), np.nan, dtype=np.float32)
                for sample, (x, y) in enumerate(gaze):
                    if not (np.isfinite([x, y]).all() and 0 <= x < 1 and 0 <= y < 1):
                        continue
                    frame = masks[min(sample // 6, len(masks) - 1)]
                    hits[sample] = frame[:, min(int(np.ceil(y * 320)), 319), min(int(np.ceil(x * 480)), 479)]
                output[f'v{video}_q{question}_u{uid}'] = hits.T
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f'Saved {len(output)} object-gaze arrays to {args.output}')


if __name__ == '__main__':
    main()
