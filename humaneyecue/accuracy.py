"""Human accuracy and response timing corresponding to Table 3 and Appendix B."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from .data import Dataset, decode, require_columns


def score(qtype, choices, truth, answer):
    """Return one score per descriptive response or per multi-choice option."""
    truth, answer, choices = decode(truth), decode(answer), decode(choices)
    if not isinstance(answer, list):
        raise ValueError(f'Expected a list of selected option indices, got {answer!r}')
    if qtype == 'descriptive:count':
        return [int(float(answer[0])) == int(truth)]
    if qtype.startswith('descriptive'):
        return [choices[int(answer[0])] == truth]
    if qtype in ('explanatory', 'predictive', 'counterfactual'):
        return [(i in answer) == (t == 'correct') for i, t in enumerate(truth)]
    raise ValueError(f'Unsupported collection question type: {qtype}')


def analyze(dataset):
    questions, sessions = dataset.questions(), dataset.sessions()
    require_columns(sessions, ['q_start', 'q_stop', 'a_start', 'a_stop'], 'sessions for timing')
    stats = defaultdict(lambda: {'correct': [], 'ttc_seconds': [], 'tta_seconds': []})
    for q in questions.to_dict('records'):
        truth = decode(q['answer'])
        if truth is None or (isinstance(truth, float) and np.isnan(truth)):
            continue
        rows = sessions[(sessions.video_no == q['video_no']) & (sessions.question_no == q['question_no'])]
        for row in rows.to_dict('records'):
            result = stats[q['question_type']]
            result['correct'].extend(score(q['question_type'], q['choices'], truth, row['answer']))
            for key, start, stop in [('ttc_seconds', 'q_start', 'q_stop'), ('tta_seconds', 'a_start', 'a_stop')]:
                seconds = (pd.Timestamp(row[stop]) - pd.Timestamp(row[start])).total_seconds()
                if not np.isfinite(seconds) or seconds < 0:
                    raise ValueError(f'Invalid {key} for video {q["video_no"]}, question {q["question_no"]}')
                result[key].append(seconds)
    if not stats:
        raise ValueError('No scored human responses found')
    result = {}
    for kind, values in sorted(stats.items()):
        result[kind] = {'accuracy_percent': 100 * float(np.mean(values['correct'])),
                        'scoring_units': len(values['correct']), 'responses': len(values['ttc_seconds']),
                        'ttc_seconds': float(np.mean(values['ttc_seconds'])),
                        'tta_seconds': float(np.mean(values['tta_seconds']))}
    units = sum((v['correct'] for v in stats.values()), [])
    result['overall'] = {'accuracy_percent': 100 * float(np.mean(units)), 'scoring_units': len(units)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='output/human_accuracy.json')
    args = parser.parse_args()
    result = analyze(Dataset(args.data_root))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
