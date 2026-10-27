"""Read the public Parquet export without a database connection."""
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd


def decode(value):
    """Accept JSON, legacy Python-literal strings, or native Parquet values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if not isinstance(value, str):
        return value
    if value in ('null', 'None', ''):
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def require_columns(frame, names, label):
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f'{label}: missing columns {sorted(missing)}')


class Dataset:
    def __init__(self, root):
        self.root = Path(root)

    def questions(self):
        frame = pd.read_parquet(self.root / 'QUESTIONS.parquet')
        require_columns(frame, ['video_no', 'question_no', 'question_type', 'answer', 'choices'], 'QUESTIONS')
        if frame.duplicated(['video_no', 'question_no']).any():
            raise ValueError('QUESTIONS contains duplicate video/question keys')
        return frame

    def sessions(self):
        choices = [self.root / 'QVA_SESSION_DATA.parquet', self.root / 'QVA_SESSION_DB.parquet']
        path = next((p for p in choices if p.exists()), choices[0])
        frame = pd.read_parquet(path)
        require_columns(frame, ['video_no', 'question_no', 'hr_uid', 'answer'], path.name)
        return frame

    def gaze_frame(self, video_no, question_no=None):
        path = self.root / 'partitioned' / f'eye_track_data_for_video_{video_no}.parquet'
        frame = pd.read_parquet(path)
        require_columns(frame, ['video_no', 'question_no', 'hr_uid', 'split_type', 'FPOGX', 'FPOGY'], path.name)
        frame = frame[(frame.video_no == video_no) & (frame.split_type == 'video')]
        if question_no is not None:
            frame = frame[frame.question_no == question_no]
        # Order independently within each participant/session. Keep invalid gaze rows:
        # deleting them would shift all subsequent 150-Hz sample indices.
        order = [c for c in ['hr_uid', 'qva_session_id', 'TICK', 'ticktime', 'TS', 'abs_time'] if c in frame]
        if order:
            frame = frame.sort_values(order, kind='stable')
        return frame

    def gaze(self, video_no, question_no):
        frame = self.gaze_frame(video_no, question_no)
        result = {}
        for uid, rows in frame.groupby('hr_uid', sort=True):
            if 'qva_session_id' in rows and rows.qva_session_id.nunique() > 1:
                raise ValueError(f'Multiple attempts for video {video_no}, question {question_no}, participant {uid}; select the study attempt before analysis')
            result[uid] = rows[['FPOGX', 'FPOGY']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        return result
