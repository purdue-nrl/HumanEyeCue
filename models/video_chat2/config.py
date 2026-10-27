"""Paper presets and local resource validation (no model dependencies)."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    name: str
    gaze: bool
    lora_lr: float
    qformer_lr: float | None
    epochs: int = 1
    frames: int = 8
    batch_size: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    threshold: float = 0.1
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.02
    warmup_epochs: float = 0.6
    min_lr_multiplier: float = 0.25
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def recipe_for(setting, action):
    if setting not in range(5):
        raise ValueError('Setting must be 0–4')
    if action not in ('train', 'evaluate'):
        raise ValueError('Action must be train or evaluate')
    if action == 'train' and setting not in (1, 4):
        raise ValueError('Train with setting 1 (VQA) or 4 (VQA+gaze)')
    gaze_trained = setting in (3, 4)
    return Recipe('vqa_gaze' if gaze_trained else 'vqa', setting in (2, 4),
                  5e-5 if gaze_trained else 2e-5, 1e-5 if gaze_trained else None)


def load_resources(filename, base_model=None):
    """All paths are local; relative paths are relative to this JSON file."""
    filename = Path(filename).resolve()
    resources = json.loads(filename.read_text(encoding='utf-8'))
    kinds = {'vit_blip_model_path': 'file', 'llama_model_path': 'dir',
             'videochat2_model_path': 'file', 'bert_model_path': 'dir'}
    for key, kind in kinds.items():
        value = base_model if key == 'llama_model_path' and base_model else resources.get(key)
        if not value:
            raise ValueError(f'Missing {key} in {filename}')
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() if key == 'llama_model_path' and base_model else filename.parent) / path
        path = path.resolve()
        if not (path.is_file() if kind == 'file' else path.is_dir()):
            raise ValueError(f'{key} must be an existing {kind}: {path}')
        resources[key] = str(path)
    return {key: resources[key] for key in kinds}


class AttrDict(dict):
    """Attribute access for the retained upstream vision builder."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def vision_config(recipe):
    return AttrDict(vision_encoder=AttrDict(
        img_size=224, patch_size=16, encoder_embed_dim=1024,
        encoder_depth=24, encoder_num_heads=16, drop_path_rate=0.,
        num_frames=recipe.frames, tubelet_size=1, use_checkpoint=False,
        checkpoint_num=0, return_index=-2, with_ln=False,
        use_mask=recipe.gaze, attn_thres=recipe.threshold, pretrained=''))
