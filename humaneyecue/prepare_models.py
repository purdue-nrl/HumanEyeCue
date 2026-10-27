"""Convert public Parquet to the CSV layout consumed by the supplied model code."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import Dataset, decode, require_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--videos-root', required=True, help='Flat directory containing selected video files')
    parser.add_argument('--output-dir', default='data/model_inputs')
    args = parser.parse_args()
    dataset = Dataset(args.data_root)
    questions = dataset.questions().copy()
    columns = ['q_id', 'video_no', 'question_no', 'question', 'choices', 'question_type', 'answer', 'num_responses', 'video', 'split']
    require_columns(questions, columns, 'model QUESTIONS export')
    split_map = {'train': 0, 'valid': 1, 'validation': 1, 'test': 2, '0': 0, '1': 1, '2': 2}
    questions['split'] = questions['split'].astype(str).map(split_map)
    if questions['split'].isna().any():
        raise ValueError('Unrecognized dataset split; do not invent or reassign the paper split')
    questions['split'] = questions['split'].astype(int)
    videos_root = Path(args.videos_root).resolve()
    for name in questions.video:
        if Path(name).name != name or not (videos_root / name).is_file():
            raise ValueError(f'Missing flat video file or invalid filename: {name}')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Legacy loaders use ast.literal_eval for these two CSV fields.
    for col in ['choices', 'answer']:
        questions[col] = questions[col].map(lambda x: 'null' if decode(x) is None else repr(decode(x)))
    questions[columns].to_csv(output / 'QUESTIONS.csv', index=False)
    for video, rows in questions.groupby('video_no'):
        frame = dataset.gaze_frame(int(video))
        dest = output / 'gaze' / f'video_no_{video}'
        dest.mkdir(parents=True, exist_ok=True)
        for question in rows.question_no:
            # Validate repeated sessions rather than concatenating separate attempts.
            dataset.gaze(int(video), int(question))
            pair = frame[frame.question_no == question].copy()
            if pair.empty:
                raise ValueError(f'No gaze for video {video}, question {question}')
            # Keep rows at invalid time steps; float('nan') is accepted by the loader.
            pair[['hr_uid', 'FPOGX', 'FPOGY']].to_csv(dest / f'video_no_{video}_question_no_{question}_video.csv', index=False, na_rep='nan')
    config = {'seed': 19, 'questions_get_from': 'csv', 'questions_csv_file': str(output / 'QUESTIONS.csv'),
              'videos_root_dir': videos_root.as_posix() + '/', 'video_frames_get_from': 'raw_videos',
              'video_frames_total': 125, 'attention_videos_fps': 25, 'attention_tracker_fps': 150,
              'attention_accumulate': True, 'attention_refractory': 3, 'attention_get_from': 'csv',
              'attention_csv_path': {'dir': str(output / 'gaze'), 'file': None}, 'sql_configs': None,
              'collection_settings': {'video_resolution': [320, 480], 'screen_resolution': [1080, 1920],
                                      'screen_diagonal_inches': 24, 'viewing_distance_cm': 55, 'visual_angle_degrees': 2}}
    (output / 'model_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Model configuration: {output / "model_config.json"}')


if __name__ == '__main__':
    main()
