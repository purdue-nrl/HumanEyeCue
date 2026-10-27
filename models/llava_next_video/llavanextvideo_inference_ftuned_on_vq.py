import os
import json
import torch
import pickle
import argparse
import numpy as np
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration
from humaneyecue_dataloader import get_humaneyecue_dataloader
from humaneyecue_helper_functions import infer_text_video_spans, text_pov_frame_maps, compute_normalized_scanpath_saliency

## Reproducibility
np.random.seed(19)
torch.manual_seed(19)
torch.cuda.manual_seed_all(19)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate(model, processor, dataloader, device, attention_query="all_text_tokens", attention_layer_idx=-1):
    model.eval()
    if attention_query in ["all_text_tokens", "pred_text_token"]:
        ## Attention weights are not computed explicitly if attn_implementation is "sdpa" or "flash"
        model.language_model.set_attn_implementation("eager")
        ## Setting up attention hook on decoder (llama model for llava-next-video)
        handles = []
        def attn_weight_hook(module, input, output, layer_idx):
            attention_weights_language_model[layer_idx].append(output[1])

        if attention_layer_idx == 'all':
            attention_weights_language_model = {i: [] for i in range(len(model.language_model.layers))}
            for i, layer in enumerate(model.language_model.layers):
                handle = layer.self_attn.register_forward_hook(lambda module, input, output, layer_idx=i: attn_weight_hook(module, input, output, layer_idx))
                handles.append(handle)
        elif isinstance(attention_layer_idx, int):
            attention_weights_language_model = {attention_layer_idx: []}
            handle = model.language_model.layers[attention_layer_idx].self_attn.register_forward_hook(lambda module, input, output, layer_idx=attention_layer_idx: attn_weight_hook(module, input, output, layer_idx))
            handles.append(handle)
        else:
            raise ValueError(f"Received unexpected value for attention_layer_idx = {attention_layer_idx}.\n")
    elif attention_query is None:
        pass
    else:
        raise ValueError
    results = []
    nss_scores = {}
    strict_acc = {question_no: {'correct': 0, 'total': 0} for question_no in [0, 1, 2, 3, 4, 6, 51, 52, 53, 54, 71, 72, 73, 74, 'all']}  # question_no: [correct, total]

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if attention_query:
                ## Reset attention buffer
                for layer_idx in attention_weights_language_model:
                    attention_weights_language_model[layer_idx].clear()

            ## Forward pass
            input_ids, attention_mask, pixel_values_videos, (vq_pairs, texts, gt_answers, human_gaze) = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values_videos=pixel_values_videos, do_sample=False)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)

            if attention_query:
                ## Analyze model attention
                ## INFO 1: llava-next-video REMOVES class embedding at the beginning of the video tokens BEFORE being processed by the language model
                ## INFO 2: default configurations preprocess video frames into 336x336 pixels dimensions, process them using kernel_size=14 and stride=14 convolution within CLIP, AND rescale video feature dimensions by 2
                ##         As a result, by default, there are (336/14)/2 = 12x12 total video tokens processed by the language model.
                ## INFO 2: attention_weights would be in the format [batch_size, num_heads, seq_length, seq_length], where seq_length is the sequence length in terms of tokens (BOTH text and video). In here, attn_weights[0, :, 4, :] would represent attention paid by token 4 (index 2) onto every other token (index 3).
                ## NOTE 3: currently, attention helper functions do not account for padding tokens, which might appear if batch_size > 1

                # Infer spans (text/video) per sample from input_ids
                spans = infer_text_video_spans(input_ids=input_ids, video_token_id=model.config.video_token_id, patches_per_frame=144, tokens_per_frame=144)
                # Convenience: per-sample frame patch indices
                frame_patch_indices_batch = [s["frame_patch_indices"] for s in spans]

                for layer_idx in attention_weights_language_model:
                    if attention_query == 'all_text_tokens':
                        # --- Case 1: all text tokens as queries ---
                        attn = attention_weights_language_model[layer_idx][0]            # [B, H, N, N]
                        B = attn.size(0)
                        frame_maps_list = []
                        for b in range(B):
                            text_idx_b = spans[b]["text_all_idx"].unsqueeze(0)  # <-- every sample will have different prompt, and hence different order of text tokens
                            maps_b = text_pov_frame_maps(
                                attn_probs=attn[b:b+1],                                 # [1, H, N, N]
                                text_query_idx=text_idx_b,                              # [1, T_b]
                                frame_patch_indices=[frame_patch_indices_batch[b]],     # list-of-list, indicating frame patch indices for this sample
                                visual_attn_shape=(12, 12),                             # spatial shape of visual attention maps per frame
                                aggregate_heads="mean",
                                aggregate_queries="mean",
                                renorm_per_frame=False
                            )
                            frame_maps_list.append(maps_b)  # list of length B; each item is [1, F, 12, 12]
                        frame_maps = torch.cat(frame_maps_list, dim=0)  # [B, F, 12, 12]
                    elif attention_query == 'pred_text_token':
                        # --- Case 2: first generated token as the single text query ---
                        attn = attention_weights_language_model[layer_idx][1]       # [B, H, P, N+P]
                        B = attn.size(0)
                        # the first generated token is index 0 along the query dimension
                        first_gen_idx = torch.zeros(B, 1, dtype=torch.long, device=attn.device)
                        frame_maps = text_pov_frame_maps(
                            attn_probs=attn,                                    # [B, H, P, N+P]
                            text_query_idx=first_gen_idx,                       # [B, 1]
                            frame_patch_indices=frame_patch_indices_batch,      # list-of-list, indicating frame patch indices for every sample
                            visual_attn_shape=(12, 12),                         # spatial shape of visual attention maps per frame
                            aggregate_heads="mean",
                            aggregate_queries="mean",
                            renorm_per_frame=False
                        )
                    ## frame_maps: [B, F, 12, 12]
                    nss_scores_batch = compute_normalized_scanpath_saliency(attention_maps=frame_maps, scanpath=human_gaze, attention_shape=(336, 336)) # original frame size applied by LLaVA-NeXT-Video preprocessing
                    if layer_idx not in nss_scores:
                        nss_scores[layer_idx] = []
                    nss_scores[layer_idx].append(nss_scores_batch)

            ## Record results
            for vq, qp, a, p in zip(vq_pairs, texts, gt_answers, preds):
                results.append({'video_no': vq[0], 'question_no': vq[1], 'prompt': qp, 'ground_truth_answer': a, 'predicted_answer': p})
                # Record strict accuracy, i.e., the model has to match the gt answer exactly.
                model_answer = p.split("ASSISTANT: Answer: ")[1]
                if len(model_answer) > 0 and model_answer.lstrip()[0] == a:
                    strict_acc[vq[1]]['correct'] += 1
                    strict_acc['all']['correct'] += 1
                strict_acc[vq[1]]['total'] += 1
                strict_acc['all']['total'] += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches")

    results.append(strict_acc)
    if attention_query:
        for handle in handles:
            handle.remove()
        for layer_idx in nss_scores:
            nss_scores[layer_idx] = torch.cat(nss_scores[layer_idx], dim=0)
            nss_avg_val = nss_scores[layer_idx].mean().item()
            nss_std_val = nss_scores[layer_idx].std().item()
            results.append({f'Avg. NSS @ {layer_idx}': nss_avg_val, f'Std. NSS @ {layer_idx}': nss_std_val})

    return results, nss_scores

def int_or_all(value):
    if value.lower() == 'all':
        return 'all'
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("attention_layer must be an integer or 'all'.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT-Video model on HumanEyeCue dataset.")
    parser.add_argument('--model_dir', type=str, default="", help='Path to LLaVA-NeXT-Video model directory')
    parser.add_argument('--model_id',  type=str, default='llava-hf/LLaVA-NeXT-Video-7B-hf', help='Id/Name of LLaVA-NeXT-Video model')
    parser.add_argument('--finetuned_model_path', type=str, default="./checkpoints/REPLACE_WITH_CHECKPOINT", help='Path to a fine-tuned model checkpoint (if any)')
    parser.add_argument('--data_cfg_file', type=str, default="data/model_inputs/model_config.json", help='Path to data config file')
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'test', 'trainval', 'all'], default='test',
                        help='Dataset split to evaluate on: train, valid, test, trainval, or all (default: test)')
    parser.add_argument('--video_sampling_method', type=str, default='uniform', help='Method for sampling video frames')
    parser.add_argument('--video_num_frames', type=int, default=16, help='Number of video frames to sample')
    parser.add_argument('--attention_type', type=str, default='spatial', choices=[None, "spatial", "temporal", "spatiotemporal"], help='Attention type to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./models/llava_next_video/results/', help='Path to save the results JSON file')
    parser.add_argument('--experiment_name', type=str, default='inference_finetuned_on_vq_test_set', help='Experiment name used in naming the output files')
    parser.add_argument('--attention_query', type=str, default='all_text_tokens', choices=['all_text_tokens', 'pred_text_token', None], help='Which text tokens to use as queries for attention analysis')
    parser.add_argument('--attention_layer', type=int_or_all, default='all', help="Layer index to extract attention weights from (default: 'all')")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_dir + args.model_id)
    processor.tokenizer.padding_side = "right"

    USE_QLORA = True
    if USE_QLORA:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        args.finetuned_model_path,
        torch_dtype = torch.float16,
        quantization_config = bnb_config if USE_QLORA else None,
    ).to(device)


    dataloader = get_humaneyecue_dataloader(humaneyecue_config_file=args.data_cfg_file,
                                         split=args.split, batch_size=args.batch_size,
                                         video_sampling_method=args.video_sampling_method, video_num_frames=args.video_num_frames,
                                         attention_type=args.attention_type,
                                         processor=processor)

    results, nss_scores = evaluate(model, processor, dataloader, device, attention_query=args.attention_query, attention_layer_idx=args.attention_layer)

    # Ensure output directory exists
    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON file
    with open(os.path.join(args.output_dir, f"{args.experiment_name}_attn_layer_{args.attention_layer}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {os.path.join(args.output_dir, f'{args.experiment_name}_attn_layer_{args.attention_layer}.json')}")

    if nss_scores is not None:
        with open(os.path.join(args.output_dir, f"{args.experiment_name}_attn_layer_{args.attention_layer}_nss.pkl"), 'wb') as f:
            pickle.dump(nss_scores, f)
        print(f"NSS scores saved to {os.path.join(args.output_dir, f'{args.experiment_name}_attn_layer_{args.attention_layer}_nss.pkl')}")

if __name__ == "__main__":
    main()
