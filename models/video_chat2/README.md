# HumanEyeCue VideoChat2

This is the curated Vicuna-7B VideoChat2 implementation for Tables 4 and 6.
It includes two one-epoch training recipes and five evaluation modes. Model
weights, adapters, datasets, and experiment outputs are not included.

## Install

Use a **separate Python 3.10 or 3.11 environment** for this model. Do not install
the shared `models/requirements.txt` here: the other models use newer
Transformers. From the repository root:

```bash
python3.11 -m venv .venv-videochat2
source .venv-videochat2/bin/activate
# Select a PyTorch wheel matching your CUDA driver. This is one example:
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r models/video_chat2/requirements.txt
```

CPU verification used Python 3.11.14, PyTorch 2.2.2+cpu, torchvision 0.17.2+cpu,
Transformers 4.40.1, PEFT 0.4.0, and timm 0.5.4. Paper-scale runs require a CUDA
GPU with memory sufficient for the unquantized 7B model. No FlashAttention,
DeepSpeed, Apex, bitsandbytes, database, or global Transformers patch is needed.
The gaze-filtering implementation is included in this model's vision encoder.

## Prepare local inputs

Prepare the public model question export and CLEVRER videos as described in the
repository README, then run:

```bash
python -m humaneyecue.prepare_models --data-root data/humaneyecue --videos-root data/clevrer/video_all_selected
```

Use the binary-expanded **model** questions, preserving supplied IDs and splits.
VideoChat2 reads the resulting `data/model_inputs/model_config.json`, question
CSV, and gaze CSVs directly. Question conversion follows the original
VideoChat2 conversion notebook: count options are `0,1,2,3`; predictive answers
select an event; explanatory/counterfactual questions append their one event and
use `correct,wrong`. Counts outside the original option vocabulary are rejected.
Other supplied answer-choice order is preserved. No examples are silently dropped
because of malformed labels. Only rows with human responses in split 0 (train)
or split 2 (test) are used.

Obtain these resources separately:

| Configuration key | Required local resource |
|---|---|
| `vit_blip_model_path` | Original `umt_l16_qformer.pth` |
| `videochat2_model_path` | Original Vicuna `videochat2_7b_stage3.pth` |
| `llama_model_path` | Vicuna **7B v0** model and tokenizer directory |
| `bert_model_path` | `bert-base-uncased` tokenizer and configuration directory |

See the [upstream VideoChat2 setup](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)
and [OpenGVLab weight listing](https://huggingface.co/OpenGVLab/videochat2/tree/main).
Follow the upstream Vicuna-v0 preparation instructions; newer Vicuna versions
are not interchangeable with the original weights.

```bash
cp models/video_chat2/model_config.example.json models/video_chat2/model_config.local.json
# Edit model_config.local.json to point to your local resources.
```

Resource paths inside that JSON resolve relative to the JSON file. The example
uses ignored `checkpoints/` directories at the repository root. No resources are
downloaded by the training/evaluation code. A `--base-model /absolute/path` override
selects a different local Vicuna directory. Local resource configurations are ignored.

## Training and evaluation

The existing launcher prints commands by default. Add `--run` to execute them:

```bash
# VQA, setting 1; final local checkpoint: output/video_chat2/train_setting_1/ckpt_00.pth
python -m humaneyecue.experiments --model video_chat2 --action train --setting 1 --model-config models/video_chat2/model_config.local.json --run

# VQA + gaze, setting 4; final local checkpoint: output/video_chat2/train_setting_4/ckpt_00.pth
python -m humaneyecue.experiments --model video_chat2 --action train --setting 4 --model-config models/video_chat2/model_config.local.json --run

# 0: original stage-3 model, no gaze filtering
python -m humaneyecue.experiments --model video_chat2 --action evaluate --setting 0 --model-config models/video_chat2/model_config.local.json --run

# 1 and 2 reuse the VQA checkpoint
python -m humaneyecue.experiments --model video_chat2 --action evaluate --setting 1 --checkpoint output/video_chat2/train_setting_1/ckpt_00.pth --model-config models/video_chat2/model_config.local.json --run
python -m humaneyecue.experiments --model video_chat2 --action evaluate --setting 2 --checkpoint output/video_chat2/train_setting_1/ckpt_00.pth --model-config models/video_chat2/model_config.local.json --run

# 3 and 4 reuse the VQA + gaze checkpoint
python -m humaneyecue.experiments --model video_chat2 --action evaluate --setting 3 --checkpoint output/video_chat2/train_setting_4/ckpt_00.pth --model-config models/video_chat2/model_config.local.json --run
python -m humaneyecue.experiments --model video_chat2 --action evaluate --setting 4 --checkpoint output/video_chat2/train_setting_4/ckpt_00.pth --model-config models/video_chat2/model_config.local.json --run
```

LoRA is installed once, then stage-3 weights are loaded, followed by the supplied
fine-tuned weights for evaluation. Original raw state dictionaries and `model`
payloads are accepted. Fine-tuned evaluation requires every trained parameter;
checkpoint shape/key mismatches fail rather than silently evaluating partial adapters.
The local checkpoint stores trainable weights and metadata, not optimizer/resume
state. Resume and distributed training are not part of this release.

Both recipes use eight frames, batch size one, one GPU, one optimizer step per
example, and one epoch. The vision encoder, vision normalization, base language
model, and projection are frozen. VQA trains only LoRA at `2e-5`. VQA+gaze trains
LoRA at `5e-5` and Q-Former plus both query-token sets at `1e-5`, with filtering
threshold `0.1`. LoRA uses rank 16, alpha 32, dropout 0.1. AdamW uses betas
`(0.9,0.999)` and weight decay 0.02, with the original bias/one-dimensional
parameter decay exclusion. No gradient clipping or quantization is enabled.
The cosine schedule uses 0.6 warm-up epochs and a minimum LR multiplier of 0.25;
the source schedule also floors the warm-up multiplier at 0.25. Seed defaults to 42.
The final one-epoch checkpoint is used without test-set model selection.

Direct entry-point options include `--seed`, `--device`, `--limit`, and `--no-nss`:

```bash
python -m models.video_chat2.run --action evaluate --setting 0 --model-config models/video_chat2/model_config.local.json --output-dir output/video_chat2/smoke --limit 1
```

`--limit` marks the run and any checkpoint as a smoke test. `--no-nss` skips
attention collection and gaze loading for unfiltered accuracy evaluation.
Filtered settings always require gaze. No-gaze settings never pass gaze to the
model, even when the evaluator loads it separately to calculate NSS.

## Results and reproduction limits

`results.json` contains accuracy in percent, per-example predictions/IDs/frame
indices, visual token counts, population mean/std statistics, and NSS for every
Q-Former cross-attention layer. `resolved_run.json` records the resources,
arguments, and recipe. Outputs and locally generated checkpoints stay outside Git.

| Setting | Training | Test filtering | Paper VQA (%) | Paper NSS |
|---|---|---|---:|---|
| 0 | Zero-shot | None | 55.94 | 0.49 ± 0.31 |
| 1 | VQA | None | 62.13 | 0.26 ± 0.25 |
| 2 | VQA | Gaze | 53.71 | 1.95 ± 0.32 |
| 3 | VQA + gaze | None | 60.40 | 0.57 ± 0.32 |
| 4 | VQA + gaze | Gaze | 64.11 | 1.95 ± 0.32 |

These are the supplied paper reference values, not measurements from this release.
The paper reports 196 visual tokens/frame without filtering and 16.9 ± 2.5 with
filtering. The evaluator measures actual retained counts; it does not enforce the
paper average. All settings produce 96 Q-Former tokens **per video**, from 32
original plus 64 extra query tokens.

NSS averages heads and queries separately for each of six cross-attention layers.
Removed spatial positions are zero-filled before calculating filtered NSS. Each
sample averages fixations within frames, then valid frames; dataset statistics
exclude samples with no valid fixations and report the valid count. Missing NSS
is JSON `null`. The paper's single-layer/aggregation choice and filtered NSS
procedure remain unverified; this evaluator does not claim numerical parity.

The source's center-crop image preprocessing and gaze saliency resize are retained:
gaze maps are resized from 320×480 to 224×224 without a matching center crop.
This inherited geometry mismatch remains a reproduction limitation. The selected
source trainer uses tensor resizing and ImageNet mean/std normalization; its
evaluator uses PIL resizing and CLIP mean/std normalization. These differences
are preserved. Training uses the source random temporal sampler; evaluation uses segment midpoints. Gaze
accumulates six 150-Hz samples per 25-Hz frame, preserving invalid sample positions.
NaN padding replaces the experimental zero padding to avoid artificial top-left
fixations. Training also uses a proper BOS embedding and explicit answer-only
loss masking. These release corrections, fixed paper hyperparameters, and the
confirmed one-epoch duration can differ from historical experimental checkpoints;
their exact selection and seeds are unverified.

Filtering with no retained tokens for an entire video fails explicitly. An empty
individual frame contributes zero retained tokens; it is not silently replaced.
Full paper-scale training and exact Table 4 reproduction have not been validated.

## Verification and provenance

```bash
python -m pip install 'pytest>=8,<10'
OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python -m pytest tests/test_video_chat2.py -q
```

The tests use synthetic videos, gaze, and tiny versions of the actual retained
backbones; they do not fetch model assets. See [THIRD_PARTY.md](THIRD_PARTY.md) for
license texts, attribution, and source provenance. The project-wide license for
the authors' contributions remains to be chosen.
