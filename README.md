# HumanEyeCue

**HumanEyeCue: A Task-Driven Dataset of Human Visual Attention Cues for Video Question Answering**<br>
Timur Ibrayev, Deepak Ravikumar, Donghun James Lee, Manish Nagaraj, Amogh Joshi, Kaushik Roy<br>
**WACV 2027**

HumanEyeCue contains task-driven eye tracking from 73 participants on 300 CLEVRER videos, with eight questions per video (2,400 video-question pairs). Participants read a question, view a video once, then answer it. The dataset supports analysis of attention consistency, question-dependent attention, and gaze-guided video question answering.

## Setup

Run commands from this repository's root unless a component README says otherwise. Use Python 3.10 or 3.11 for a fresh installation:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Model experiments additionally need a CUDA GPU and an appropriate CUDA PyTorch installation. For Video-LLaVA and LLaVA-NeXT-Video, install `models/requirements.txt`; the original environment used PyTorch 2.8.0/CUDA 12.6 and Transformers 4.56.1. Their gaze-guided settings require the authors' custom Transformers token-filtering patch. [VideoChat2](models/video_chat2/README.md) uses a separate environment with its own pinned requirements and includes its gaze-filtering implementation. No model checkpoints are distributed. [Capture](capture/README.md) has separate Windows/hardware dependencies.

## Dataset layout

Download the public database export: **[Dataset download coming soon](https://example.com/)**. This link will be replaced with the release URL. The examples read the exported Parquet tables locally; no MySQL server is needed.

Extract the export into `data/humaneyecue/` and place the matching CLEVRER videos under `data/clevrer/`, as below. Keep dataset files outside Git:

```text
data/
  humaneyecue/
    QUESTIONS.parquet
    QVA_SESSION_DATA.parquet
    partitioned/
      eye_track_data_for_video_0.parquet
      ...
  clevrer/
    video_all_selected/
      video_00047.mp4
      ...
    qa_and_parsed_programs_all_selected.json       # object analysis only
    processed_proposals_all_selected/             # object analysis only
```

`QVA_SESSION_DB.parquet` is accepted as a legacy alias. Video IDs in the study are selected-dataset IDs; they must not be assumed to equal the original CLEVRER scene indices. Preserve the supplied `video` filename and split mapping.

| Export table | Required columns |
| --- | --- |
| `QUESTIONS.parquet` | `video_no`, `question_no`, `question_type`, `choices`, `answer` |
| `QVA_SESSION_DATA.parquet` | `video_no`, `question_no`, `hr_uid`, `answer`; accuracy/timing also needs `q_start`, `q_stop`, `a_start`, `a_stop` |
| Per-video gaze Parquet | `video_no`, `question_no`, `hr_uid`, `split_type`, `FPOGX`, `FPOGY`; preserve `qva_session_id` and sample order (`TICK`/`TS` or `ticktime`/`abs_time`) when available |

Model conversion also needs `q_id`, `question`, `num_responses`, `video`, and `split` in the question table. Human accuracy uses the original collection questions; model runs use the binary-expanded questions. Keep those exports in separate data roots if provided separately. Registration names and email addresses are not part of the public export.

## Use the dataset

After setup, use the Python API to read questions, participant responses, and gaze for one recorded video/question pair:

```python
from humaneyecue.data import Dataset

dataset = Dataset("data/humaneyecue")
questions = dataset.questions()       # pandas DataFrame of question metadata
responses = dataset.sessions()       # pandas DataFrame of participant responses
trial = responses.iloc[0]            # choose a recorded pair from your export
video_no, question_no = int(trial.video_no), int(trial.question_no)
question = questions[
    (questions.video_no == video_no) & (questions.question_no == question_no)
]
print(question.to_dict("records"))

gaze_by_user = dataset.gaze(video_no, question_no)
for participant_id, xy in gaze_by_user.items():
    print(participant_id, xy.shape)   # (number of samples, 2): normalized x, y
```

Coordinates use a top-left origin and are sampled at 150 Hz. Invalid samples retain their positions in the sequence. `gaze()` selects the video segment; `gaze_frame()` also exposes its table columns. Use a pair with recorded gaze, and preserve the export's trial and video mapping.

### Visualization example

[examples/example_visualization_code.py](examples/example_visualization_code.py) adapts the original source example to the `Dataset` API. It draws each participant's gaze in a different color, repeats each 25-Hz video frame six times to render at 150 Hz, and keeps remaining gaze samples on the last frame. It skips invalid coordinates. Read or import `render_visualization()` to adapt it for your own analysis.

```bash
python -m examples.example_visualization_code --data-root data/humaneyecue --video-no 0 --question-no 6 --video data/clevrer/video_all_selected/video_00047.mp4 --output output/example_visualization.mp4
```

The IDs and filename above are illustrative: replace them with a recorded pair and its matching video from your export. For a compact overlay at the video's native frame rate, use `python -m humaneyecue.visualize` with the same arguments.

## Run the paper's experiments

```bash
# Table 3: human accuracy; Appendix B: question/answer timing.
python -m humaneyecue.accuracy --data-root data/humaneyecue

# Figure 2, Sections 4.2/4.3: intra/inter-question NSS and random control.
# Start with one selected ID that exists in your export; omit --videos for all IDs.
python -m humaneyecue.attention --data-root data/humaneyecue --videos 0
python -m humaneyecue.attention --data-root data/humaneyecue

# Attention-video examples. Supply the matching original CLEVRER video explicitly.
python -m humaneyecue.visualize --data-root data/humaneyecue --video-no 0 --question-no 6 --video data/clevrer/video_all_selected/video_00047.mp4

# Optional object-level analysis: install pycocotools and obtain processed masks.
python -m pip install pycocotools
python -m humaneyecue.objects --data-root data/humaneyecue --clevrer-root data/clevrer

# Convert Parquet exports into the original model loaders' local CSV layout.
python -m humaneyecue.prepare_models --data-root data/humaneyecue --videos-root data/clevrer/video_all_selected

# Resolve a Table 4 command without loading any model or starting a job.
python -m humaneyecue.experiments --model video_llava --action train --setting 1
# Add --run to execute in a configured GPU environment.
python -m humaneyecue.experiments --model video_llava --action train --setting 1 --run
```

`prepare_models` writes `data/model_inputs/model_config.json`, which the model launcher and individual model scripts use by default. `video_llava`, `llava_next_video`, and `video_chat2` are supported. Settings are: `0` zero-shot, `1` VQA fine-tuned, `2` VQA fine-tuned with gaze filtering, `3` VQA+gaze fine-tuned without filtering, and `4` VQA+gaze fine-tuned with filtering. Train with setting `1` or `4`; evaluate with `--action evaluate --setting N`, adding `--checkpoint /path/to/checkpoint` for settings `1`–`4`. VideoChat2 accepts a checkpoint file plus `--model-config` for external pretrained resources; see its [setup and all five evaluation commands](models/video_chat2/README.md). Settings `2` and `4` require the custom Transformers gaze patch for the two LLaVA models; VideoChat2 includes its own filtering code.

Results go to `output/`. The human tools use local Parquet, not the collection database. Full attention analysis is CPU/memory intensive (one 900-sample sequence of 320x480 saliency maps uses roughly 0.55 GB before Python overhead); use `--videos` to split work. The random control uses seed 19. The new offline implementation preserves missing sample positions as NaN rather than inserting synthetic top-left fixations; confirm numerical parity against the authors' final exports before claiming exact reproduction.

Original plotting notebooks are retained under `analysis/notebooks/` as references with outputs cleared; their legacy SQL/path cells require adaptation and are not the public entry points.

## Citation

Timur Ibrayev, Deepak Ravikumar, Donghun James Lee, Manish Nagaraj, Amogh Joshi, and Kaushik Roy. *HumanEyeCue: A Task-Driven Dataset of Human Visual Attention Cues for Video Question Answering.* **WACV 2027**.

```bibtex
@inproceedings{ibrayev2027humaneyecue,
  title     = {HumanEyeCue: A Task-Driven Dataset of Human Visual Attention Cues for Video Question Answering},
  author    = {Ibrayev, Timur and Ravikumar, Deepak and Lee, Donghun James and Nagaraj, Manish and Joshi, Amogh and Roy, Kaushik},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```
