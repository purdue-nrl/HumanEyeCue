"""Intra/inter-question NSS (Sections 4.2/4.3), with temporal correlation."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import Dataset
from .legacy_metrics import calculate_ppd, create_normalized_saliency_maps


def scanpath_scores(fixations, maps):
    """Keep missing time steps as NaN so temporal correlations stay aligned."""
    result = np.full(len(maps), np.nan)
    for t, (xy, saliency) in enumerate(zip(fixations, maps)):
        x, y = xy
        if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < 1 and 0 <= y < 1):
            continue
        std = float(saliency.std())
        result[t] = 0.0 if std == 0 else (saliency[int(y * saliency.shape[0]), int(x * saliency.shape[1])] - saliency.mean()) / std
    return result


def mean_valid(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', default='output/attention')
    parser.add_argument('--videos', type=int, nargs='+', help='Selected dataset IDs; default: all available IDs')
    parser.add_argument('--samples', type=int, default=900)
    parser.add_argument('--visual-angle-deg', type=float, default=2)
    parser.add_argument('--seed', type=int, default=19)
    args = parser.parse_args()
    if args.samples <= 0 or args.visual_angle_deg <= 0:
        parser.error('samples and visual-angle-deg must be positive')
    dataset = Dataset(args.data_root)
    questions = dataset.questions()
    available = set(questions.video_no.astype(int))
    videos = sorted(available) if args.videos is None else sorted(set(args.videos))
    if not set(videos) <= available:
        parser.error('Requested video IDs are absent from QUESTIONS.parquet')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sigma = calculate_ppd(visual_angle_deg=args.visual_angle_deg, screen_diagonal_inches=24) * 480 / 1920
    rng = np.random.default_rng(args.seed)
    records, correlations = [], []
    for video in videos:
        # Load one video at a time. Build saliency for a comparison question on demand
        # rather than holding all 8 x 900 x 320 x 480 maps in memory.
        gaze = {}
        for question in sorted(map(int, questions.loc[questions.video_no == video, 'question_no'].unique())):
            users = dataset.gaze(video, question)
            if not users:
                continue
            data = np.full((len(users), args.samples, 2), np.nan, dtype=np.float32)
            for i, values in enumerate(users.values()):
                count = min(len(values), args.samples)
                data[i, :count] = values[:count]
            gaze[question] = (list(users), data)
        temporal = {}
        for question, (uids, data) in gaze.items():
            series = []
            for i, uid in enumerate(uids):
                if len(uids) < 2:
                    continue
                maps = create_normalized_saliency_maps(np.delete(data, i, axis=0), sigma=sigma)
                scores = scanpath_scores(data[i], maps)
                series.append(scores)
                records.append({'kind': 'intra', 'video_no': video, 'question_no': question,
                                'hr_uid': int(uid), 'reference_question_no': question, 'nss': mean_valid(scores)})
                del maps
            if series:
                matrix = np.asarray(series)
                counts = np.isfinite(matrix).sum(axis=0)
                temporal[question] = np.divide(np.nansum(matrix, axis=0), counts,
                                               out=np.full(args.samples, np.nan), where=counts > 0)
        for reference, (_, reference_data) in gaze.items():
            maps = create_normalized_saliency_maps(reference_data, sigma=sigma)
            for question, (uids, data) in gaze.items():
                if question == reference:
                    continue
                for uid, scanpath in zip(uids, data):
                    records.append({'kind': 'inter', 'video_no': video, 'question_no': question,
                                    'hr_uid': int(uid), 'reference_question_no': reference,
                                    'nss': mean_valid(scanpath_scores(scanpath, maps))})
            for run in range(3):
                records.append({'kind': 'random', 'video_no': video, 'question_no': reference,
                                'run': run, 'nss': mean_valid(scanpath_scores(rng.uniform(size=(args.samples, 2)), maps))})
            del maps
        for q1, s1 in temporal.items():
            for q2, s2 in temporal.items():
                valid = np.isfinite(s1) & np.isfinite(s2)
                corr = None
                if valid.sum() >= 2 and np.std(s1[valid]) > 0 and np.std(s2[valid]) > 0:
                    corr = float(np.corrcoef(s1[valid], s2[valid])[0, 1])
                correlations.append({'video_no': video, 'question_1': q1, 'question_2': q2, 'pearson_r': corr})
        np.savez_compressed(output / f'temporal_video_{video}.npz', **{str(k): v for k, v in temporal.items()})
        print(f'Analyzed video {video}', flush=True)
    if not records:
        raise ValueError('No gaze comparisons found')
    report = {'settings': vars(args), 'sigma_pixels': float(sigma), 'records': records,
              'temporal_correlations': correlations,
              'means': {kind: mean_valid([r['nss'] for r in records if r['kind'] == kind and r['nss'] is not None])
                        for kind in ('intra', 'inter', 'random')}}
    (output / 'nss.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind, color in [('inter', '#347c9c'), ('intra', '#b95343')]:
        values = [r['nss'] for r in records if r['kind'] == kind and r['nss'] is not None]
        if values:
            plt.hist(values, bins=40, density=True, alpha=0.6, label=kind, color=color)
    plt.xlabel('Normalized scanpath saliency (NSS)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / 'nss_distribution.png', dpi=180)
    plt.close()
    print(json.dumps(report['means']))


if __name__ == '__main__':
    main()
