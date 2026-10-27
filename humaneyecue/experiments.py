"""Print or execute an explicit Table 4 model experiment from the release root."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['video_llava', 'llava_next_video', 'video_chat2'], required=True)
    parser.add_argument('--action', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--setting', type=int, choices=range(5), required=True)
    parser.add_argument('--data-config', default='data/model_inputs/model_config.json')
    parser.add_argument('--checkpoint', help='Saved VQA checkpoint for settings 1/2; saved gaze checkpoint for settings 3/4')
    parser.add_argument('--base-model', help='Override Hugging Face identifier with a local model path')
    parser.add_argument('--model-config', help='VideoChat2 JSON with local pretrained resource paths')
    parser.add_argument('--output-dir')
    parser.add_argument('--run', action='store_true', help='Without this flag, only print the resolved command')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == 'train' and args.setting not in (1, 4):
        parser.error('Train VQA with setting 1 or VQA+gaze with setting 4; settings 2/3 reuse these checkpoints')
    if args.action == 'evaluate' and args.setting > 0 and not args.checkpoint:
        parser.error('--checkpoint is required for fine-tuned evaluation')
    gaze = args.setting in (2, 4)
    config = str(Path(args.data_config).resolve())
    destination = str(Path(args.output_dir or f'output/{args.model}/{args.action}_setting_{args.setting}').resolve())
    if args.model == 'video_chat2':
        command = [sys.executable, '-m', 'models.video_chat2.run', '--action', args.action,
                   '--setting', str(args.setting), '--data-config', config, '--output-dir', destination,
                   '--model-config', str(Path(args.model_config or root / 'models/video_chat2/model_config.example.json').resolve())]
        if args.base_model:
            command += ['--base-model', str(Path(args.base_model).resolve())]
        if args.checkpoint:
            command += ['--checkpoint', str(Path(args.checkpoint).resolve())]
        print(subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command), flush=True)
        if args.run:
            from models.video_chat2.run import parser as videochat_parser, preflight
            try:
                preflight(videochat_parser().parse_args(command[3:]))
            except (OSError, ValueError, KeyError) as exc:
                parser.error(str(exc))
            subprocess.run(command, cwd=root, check=True)
        return
    if args.model_config:
        parser.error('--model-config is only supported for VideoChat2')
    prefix, frames, model_id = {'video_llava': ('videollava', 8, 'LanguageBind/Video-LLaVA-7B-hf'),
                                'llava_next_video': ('llavanextvideo', 16, 'llava-hf/LLaVA-NeXT-Video-7B-hf')}[args.model]
    if args.action == 'train':
        suffix = '_finetuning_on_vq_and_gaze' if gaze else '_finetuning_on_vq'
    else:
        suffix = '_inference_zeroshot' if args.setting == 0 else ('_inference_ftuned_on_vq_and_gaze' if gaze else '_inference_ftuned_on_vq')
    script = root / 'models' / args.model / (prefix + suffix + '.py')
    command = [sys.executable, str(script)]
    if args.action == 'train':
        command += ['--model', args.base_model or model_id, '--data-config', config, '--output-dir', destination]
    else:
        command += ['--model_dir', '', '--model_id', args.base_model or model_id, '--data_cfg_file', config,
                    '--split', 'test', '--video_num_frames', str(frames), '--batch_size', '1',
                    '--attention_type', 'spatial', '--output_dir', destination, '--experiment_name', f'setting_{args.setting}']
        if args.model != 'video_llava' or args.setting != 0:
            command += ['--attention_layer', '0']
        if args.checkpoint:
            command += ['--finetuned_model_path', str(Path(args.checkpoint).resolve())]
    print(subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command), flush=True)
    if gaze:
        print('REQUIRES authors\' custom Transformers token-filtering implementation.', flush=True)
    if args.run:
        if not Path(config).is_file():
            parser.error('Missing data configuration; run python -m humaneyecue.prepare_models first')
        if args.checkpoint and not Path(args.checkpoint).is_dir():
            parser.error('Checkpoint directory does not exist')
        if gaze:
            # The supplied public dependency does not contain these methods. Fail
            # before downloading/loading multi-billion-parameter weights.
            import inspect
            from transformers.models.clip.modeling_clip import CLIPVisionTransformer
            if not all(hasattr(CLIPVisionTransformer, key) for key in ('store_human_attention', 'clear_human_attention')):
                parser.error('Custom CLIP gaze-filtering patch is missing; stock Transformers cannot run this setting')
        Path(destination).mkdir(parents=True, exist_ok=True)
        (Path(destination) / 'release_run.json').write_text(json.dumps({'arguments': vars(args), 'command': command}, indent=2), encoding='utf-8')
        subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
