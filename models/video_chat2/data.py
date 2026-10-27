"""Lazy video loading from humaneyecue.prepare_models' CSV export."""
import csv
import json
from pathlib import Path
import random

import numpy as np

from humaneyecue.data import decode


SYSTEM = ('Carefully watch the video and pay attention to the cause and sequence of events, '
          'the detail and movement of objects, and the action and pose of persons. Based on '
          'your observations, select the best option that accurately addresses the question.\n')


def qa_text(row):
    choices = decode(row['choices'])
    answer = decode(row['answer'])
    question = row['question']
    kind = row.get('question_type', '')
    # Collection exports store event descriptions in choices, whereas descriptive
    # answer vocabularies are implicit. Match the original VideoChat2 JSON format.
    vocabularies = {
        'descriptive:count': ['0', '1', '2', '3'],
        'descriptive:exist': ['yes', 'no'],
        'descriptive:query_material': ['metal', 'rubber'],
        'descriptive:query_shape': ['cube', 'sphere', 'cylinder'],
        'descriptive:query_color': ['gray', 'red', 'blue', 'green', 'brown', 'cyan', 'purple', 'yellow'],
    }
    if kind in vocabularies:
        if kind == 'descriptive:count' or not choices:
            choices = vocabularies[kind]
        answer = str(answer)
    elif kind == 'predictive' and isinstance(answer, list):
        if not isinstance(choices, list) or len(choices) != len(answer) or answer.count('correct') != 1:
            raise ValueError('Predictive questions must contain exactly one correct event')
        answer = choices[answer.index('correct')]
    elif 'explanatory' in kind or 'counterfactual' in kind:
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError('Use binary-expanded questions with one event per row')
        question = question.rstrip() + ' ' + str(choices[0])
        choices = ['correct', 'wrong']
    if not isinstance(choices, list) or not choices or len(choices) > 26:
        raise ValueError('Expected 1–26 ordered answer choices')
    if isinstance(answer, list):
        if len(answer) != 1:
            raise ValueError('Use the binary-expanded model question export; expected one answer')
        answer = answer[0]
    if answer not in choices:
        raise ValueError(f'Answer {answer!r} is absent from choices for question {row["question_no"]}')
    lines = [f'Question: {question}', 'Options:']
    lines += [f'({chr(65+i)}) {choice}' for i, choice in enumerate(choices)]
    option = chr(65 + choices.index(answer))
    return '\n'.join(lines), f'({option}) {answer}', option


def training_text(question, answer):
    return (f'###Human: <Video></Video> ###Human: {question} ###Assistant: ',
            f'Answer: {answer} ###')


def evaluation_text(question):
    return (SYSTEM + '###Human: <Video><VideoHere></Video>\n###Human: '
            + SYSTEM + question + '\nOnly give the best option.\n###Assistant: Best option:(')


def frame_indices(length, training=False, frames=8):
    if length < 1:
        raise ValueError('Video contains no frames')
    if not training:
        segment = (length - 1) / frames
        return np.clip([int(segment / 2 + np.round(segment*i)) for i in range(frames)],
                       0, length - 1).astype(int)
    # Original training sampler: random positions in uniform temporal intervals.
    count = min(frames, length)
    boundaries = np.linspace(0, length, count + 1).astype(int)
    ranges = [(int(a), int(b)-1) for a, b in zip(boundaries[:-1], boundaries[1:])]
    if all(b > a for a, b in ranges):
        indices = [random.choice(range(a, b)) for a, b in ranges]
    else:
        indices = sorted(np.random.permutation(length)[:count].tolist())
    return np.array(indices + [indices[-1]] * (frames-len(indices)))


def read_gaze(path, indices, tracker_fps=150, video_fps=25):
    ratio = tracker_fps / video_fps
    if ratio != int(ratio) or ratio < 1:
        raise ValueError('Tracker FPS must be an integer multiple of video FPS')
    recordings = {}
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            recordings.setdefault(row['hr_uid'], []).append((float(row['FPOGX']), float(row['FPOGY'])))
    result = []
    for index in indices:
        start = int(index * ratio)
        result.append([point for recording in recordings.values()
                       for point in recording[start:start+int(ratio)]])
    return result


def pad_gaze(gaze):
    padded = np.full((len(gaze), max(1, max(map(len, gaze))), 2), np.nan, dtype=np.float32)
    for i, points in enumerate(gaze):
        if points:
            padded[i, :len(points)] = points
    return padded


class VideoDataset:
    def __init__(self, config_file, training=False, need_gaze=True):
        config_file = Path(config_file).resolve()
        self.config = json.loads(config_file.read_text(encoding='utf-8'))
        self.training, self.need_gaze = training, need_gaze

        def resolve(value):
            path = Path(value).expanduser()
            return path if path.is_absolute() else config_file.parent / path

        self.videos = resolve(self.config['videos_root_dir'])
        self.gaze_dir = resolve(self.config['attention_csv_path']['dir']) if need_gaze else None
        with resolve(self.config['questions_csv_file']).open(newline='', encoding='utf-8') as stream:
            self.rows = [row for row in csv.DictReader(stream)
                         if int(row['split']) == (0 if training else 2) and int(row['num_responses']) > 0]
        if not self.rows:
            raise ValueError(f'No {"train" if training else "test"} questions in data configuration')
        self.keys = set()
        # Validate filenames and labels before loading any model weights.
        for row in self.rows:
            key = (row['video_no'], row['question_no'])
            if key in self.keys:
                raise ValueError(f'Duplicate video/question identifier: {key}')
            self.keys.add(key)
            qa_text(row)
            name = row['video']
            if Path(name).name != name or not (self.videos / name).is_file():
                raise ValueError(f'Missing video or invalid filename: {name}')
            if need_gaze and not self.gaze_path(row).is_file():
                raise ValueError(f'Missing gaze CSV: {self.gaze_path(row)}')

    def gaze_path(self, row):
        video, question = row['video_no'], row['question_no']
        return self.gaze_dir / f'video_no_{video}' / f'video_no_{video}_question_no_{question}_video.csv'

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        from decord import VideoReader, cpu
        from PIL import Image
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode

        row = self.rows[index]
        reader = VideoReader(str(self.videos / row['video']), ctx=cpu(0), num_threads=1)
        indices = frame_indices(len(reader), self.training)
        images = [Image.fromarray(reader[int(i)].asnumpy()).convert('RGB') for i in indices]
        if self.training:
            # The selected trainer used tensor resizing and ImageNet normalization;
            # its evaluation loader used PIL resizing and CLIP normalization.
            frames = torch.stack([torch.from_numpy(np.array(image)).permute(2, 0, 1) for image in images])
            frames = T.Resize(224, interpolation=InterpolationMode.BICUBIC)(frames)
            frames = T.CenterCrop(224)(frames).float().div(255.)
            frames = T.Normalize([.485, .456, .406], [.229, .224, .225])(frames)
        else:
            transform = T.Compose([T.Resize(224, interpolation=InterpolationMode.BICUBIC),
                                   T.CenterCrop(224), T.ToTensor(),
                                   T.Normalize([0.48145466, 0.4578275, 0.40821073],
                                               [0.26862954, 0.26130258, 0.27577711])])
            frames = torch.stack([transform(image) for image in images])
        question, answer, option = qa_text(row)
        gaze = read_gaze(self.gaze_path(row), indices,
                         self.config.get('attention_tracker_fps', 150),
                         self.config.get('attention_videos_fps', 25)) if self.need_gaze else None
        return {'video': frames, 'gaze': gaze, 'frame_indices': indices.tolist(),
                'question': question, 'answer': answer, 'option': option,
                'video_no': row['video_no'], 'question_no': row['question_no']}
