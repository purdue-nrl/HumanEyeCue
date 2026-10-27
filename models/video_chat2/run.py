"""Train or evaluate the paper's VideoChat2 settings. Use --help without CUDA."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import re

from .config import load_resources, recipe_for


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--action', choices=['train', 'evaluate'], required=True)
    result.add_argument('--setting', type=int, choices=range(5), required=True)
    result.add_argument('--data-config', default='data/model_inputs/model_config.json')
    result.add_argument('--model-config', default='models/video_chat2/model_config.example.json')
    result.add_argument('--checkpoint')
    result.add_argument('--base-model', help='Local Vicuna path overriding the model configuration')
    result.add_argument('--output-dir', required=True)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--limit', type=int, help='Smoke test only: stop after this many examples')
    result.add_argument('--no-nss', action='store_true', help='Evaluate accuracy without loading gaze for NSS')
    return result


def preflight(args):
    """Local paths and settings, checked before importing the model stack."""
    recipe = recipe_for(args.setting, args.action)
    if args.action == 'evaluate' and args.setting > 0 and not args.checkpoint:
        raise ValueError('--checkpoint is required for fine-tuned evaluation')
    if args.setting == 0 and args.checkpoint:
        raise ValueError('Zero-shot evaluation must not load a fine-tuned checkpoint')
    if args.action == 'train' and args.checkpoint:
        raise ValueError('Training starts from stage-3 weights; resume is not supported')
    if args.checkpoint and not Path(args.checkpoint).is_file():
        raise ValueError(f'Checkpoint file does not exist: {args.checkpoint}')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if not Path(args.data_config).is_file():
        raise ValueError('Missing data configuration; run python -m humaneyecue.prepare_models first')
    return recipe, load_resources(args.model_config, args.base_model)


def parse_option(prediction):
    match = re.match(r'^\s*\(?([A-Z])\s*[).]', prediction, re.IGNORECASE)
    return match.group(1).upper() if match else None


def summary(values):
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {'mean': None, 'std': None, 'count': 0}
    mean = sum(finite) / len(finite)
    return {'mean': mean, 'std': math.sqrt(sum((v-mean)**2 for v in finite)/len(finite)), 'count': len(finite)}


def train(model, dataset, args, recipe, destination):
    import torch
    from .data import pad_gaze, training_text
    from .optimization import configure_trainable, optimizer_groups, lr_multiplier

    configure_trainable(model, recipe)
    model.train()
    optimizer = torch.optim.AdamW(optimizer_groups(model, recipe), betas=recipe.betas)
    count = min(len(dataset), args.limit or len(dataset))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, count, recipe))
    scaler = torch.cuda.amp.GradScaler(enabled=str(args.device).startswith('cuda'))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    losses = []
    for step, index in enumerate(indices[:count]):
        example = dataset[index]
        video = example['video'].unsqueeze(0).to(args.device)
        gaze = torch.from_numpy(pad_gaze(example['gaze'])).unsqueeze(0).to(args.device) if recipe.gaze else None
        prefix, answer = training_text(example['question'], example['answer'])
        optimizer.zero_grad(set_to_none=True)
        loss = model(video, prefix, answer, gaze)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at step {step}, video {example["video_no"]}, question {example["question_no"]}')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(float(loss.detach()))
        print(f'{step+1}/{count} loss={losses[-1]:.5f}', flush=True)
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {name: value.detach().cpu() for name, value in model.state_dict().items() if name in names}
    torch.save({'model': state, 'recipe': recipe.to_dict(), 'training_setting': args.setting,
                'epoch': 0, 'global_step': count, 'seed': args.seed, 'smoke_test': args.limit is not None},
               destination / 'ckpt_00.pth')
    return {'examples': count, 'loss': summary(losses), 'checkpoint': str(destination / 'ckpt_00.pth')}


def evaluate(model, dataset, args, recipe):
    import torch
    from .attention import restore_attention, compute_normalized_scanpath_saliency
    from .data import pad_gaze, evaluation_text

    model.eval()
    layers = [layer.crossattention.self for layer in model.qformer.bert.encoder.layer
              if getattr(layer, 'has_cross_attention', False)]
    for layer in layers:
        layer.save_attention = not args.no_nss
    rows, counts = [], []
    nss = {str(i): [] for i in range(len(layers))} if not args.no_nss else {}
    for index in range(min(len(dataset), args.limit or len(dataset))):
        example = dataset[index]
        for layer in layers:
            layer.attention_map = None
        video = example['video'].unsqueeze(0).to(args.device)
        gaze = torch.from_numpy(pad_gaze(example['gaze'])).unsqueeze(0).to(args.device) if recipe.gaze else None
        prediction = model.predict(video, evaluation_text(example['question']), gaze)
        row = {key: example[key] for key in ('video_no', 'question_no', 'frame_indices', 'option')}
        row.update(prediction=prediction, correct=parse_option(prediction) == example['option'])
        row['visual_tokens_per_frame'] = model.last_token_counts.cpu().tolist()
        counts.extend(row['visual_tokens_per_frame'])
        if not args.no_nss:
            row['nss_per_layer'] = {}
            for i, layer in enumerate(layers):
                attention = layer.get_attention_map()
                if attention is None:
                    raise RuntimeError(f'Missing attention map for Q-Former cross-attention layer {i}')
                keep = model.last_keep_indices[0] if recipe.gaze else None
                spatial = restore_attention(attention, keep, recipe.frames)
                score = compute_normalized_scanpath_saliency(spatial, [example['gaze']], (14, 14)).item()
                nss[str(i)].append(score)
                row['nss_per_layer'][str(i)] = score if math.isfinite(score) else None
        rows.append(row)
        print(f'{index+1}/{len(dataset)} correct={row["correct"]}', flush=True)
    return {'examples': len(rows), 'accuracy_percent': 100*sum(row['correct'] for row in rows)/len(rows),
            'visual_tokens_per_frame': summary(counts), 'qformer_tokens_per_video': 96,
            'nss_per_layer': {key: summary(values) for key, values in nss.items()},
            'nss_convention': 'Mean heads/queries; removed patches zero-filled. Paper single-value aggregation unverified.',
            'predictions': rows}


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        recipe, resources = preflight(args)
        recipe = replace(recipe, seed=args.seed)
        from .data import VideoDataset
        dataset = VideoDataset(args.data_config, args.action == 'train',
                               recipe.gaze or (args.action == 'evaluate' and not args.no_nss))
    except (OSError, ValueError, KeyError) as exc:
        cli.error(str(exc))
    try:
        import numpy as np
        import torch
        from .model import VideoChat2, load_weights
    except ImportError as exc:
        cli.error(f'Missing VideoChat2 dependency ({exc}); install models/video_chat2/requirements.txt in a separate environment')
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        cli.error('CUDA is unavailable; run model experiments in a configured GPU environment')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model = VideoChat2(resources, recipe)
    if args.checkpoint:
        # Require every component trained by this setting, including extra queries.
        from .optimization import configure_trainable
        configure_trainable(model, recipe)
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        payload = load_weights(model, args.checkpoint, allowed_missing=names)
        if 'training_setting' in payload and payload['training_setting'] != (4 if args.setting in (3, 4) else 1):
            cli.error('Checkpoint training setting does not match evaluation setting')
    model.to(args.device)
    metadata = {'arguments': vars(args), 'recipe': recipe.to_dict(), 'resources': resources,
                'smoke_test': args.limit is not None}
    (destination / 'resolved_run.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    result = train(model, dataset, args, recipe, destination) if args.action == 'train' else evaluate(model, dataset, args, recipe)
    (destination / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
