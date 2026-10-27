import os
import json
import torch
import pickle
import argparse
from transformers import VideoLlavaForConditionalGeneration, AutoProcessor
from humaneyecue_dataloader import get_humaneyecue_dataloader
from humaneyecue_helper_functions import infer_text_video_spans, text_pov_frame_maps, compute_normalized_scanpath_saliency


def evaluate(model, processor, dataloader, device, attention_query="all_text_tokens"):
    model.eval()
    if attention_query in ["all_text_tokens", "pred_text_token"]:
        ## Attention weights are not computed explicitly if attn_implementation is "sdpa" or "flash"
        model.language_model.set_attn_implementation("eager")
        ## Setting up attention hook on decode (llama model for video-llava)
        attention_weights_llama = []
        def attn_weight_hook(module, input, output):
            attention_weights_llama.append(output[1])
        handle = model.language_model.layers[0].self_attn.register_forward_hook(attn_weight_hook)
    elif attention_query is None:
        pass
    else:
        raise ValueError
    results = []
    nss_scores = []
    strict_acc = {question_no: {'correct': 0, 'total': 0} for question_no in [0, 1, 2, 3, 4, 6, 51, 52, 53, 54, 71, 72, 73, 74, 'all']}  # question_no: [correct, total]

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if attention_query:
                ## Reset attention buffer
                attention_weights_llama.clear()

            ## Forward pass
            input_ids, attention_mask, pixel_values_videos, (vq_pairs, texts, gt_answers, human_gaze) = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values_videos=pixel_values_videos)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)

            if attention_query:
                ## Analyze model attention
                ## INFO 1: video-llava appends class embedding AT THE BEGINNING of the video tokens, i.e., if the dimension is 257, then 0 is for class embedding and 256 are for tokens of each frame with 16x16 patches. (found in CLIPVisionEmbeddings' forward function)
                ## INFO 2: attn_weights would be in the format [batch_size, num_heads, seq_length, seq_length], where seq_length is the sequence length in terms of tokens (both text and video). In here, attn_weights[0, :, 4, :] would represent attention paid by token 4 (index 2) onto every other token (index 3).
                ## NOTE 3: currently, attention helper functions do not account for padding tokens, which might appear if batch_size > 1

                # Infer spans (text/video) per sample from input_ids
                spans = infer_text_video_spans(input_ids=input_ids, video_token_id=model.config.video_token_id, patches_per_frame=256, tokens_per_frame=257)
                # Convenience: per-sample frame patch indices
                frame_patch_indices_batch = [s["frame_patch_indices"] for s in spans]

                if attention_query == 'all_text_tokens':
                    # --- Case 1: all text tokens as queries ---
                    attn = attention_weights_llama[0]            # [B, H, N, N]
                    B = attn.size(0)
                    frame_maps_list = []
                    for b in range(B):
                        text_idx_b = spans[b]["text_all_idx"].unsqueeze(0)  # <-- every sample will have different prompt, and hence different order of text tokens
                        maps_b = text_pov_frame_maps(
                            attn_probs=attn[b:b+1],                                 # [1, H, N, N]
                            text_query_idx=text_idx_b,                              # [1, T_b]
                            frame_patch_indices=[frame_patch_indices_batch[b]],     # list-of-list, indicating frame patch indices for this sample
                            aggregate_heads="mean",
                            aggregate_queries="mean",
                            renorm_per_frame=False
                        )
                        frame_maps_list.append(maps_b)  # list of length B; each item is [1, F, 16, 16]
                    frame_maps = torch.cat(frame_maps_list, dim=0)  # [B, F, 16, 16]

                elif attention_query == 'pred_text_token':
                    # --- Case 2: first generated token as the single text query ---
                    attn = attention_weights_llama[1]       # [B, H, P, N+P]
                    B = attn.size(0)
                    # the first generated token is index 0 along the query dimension
                    first_gen_idx = torch.zeros(B, 1, dtype=torch.long, device=attn.device)
                    frame_maps = text_pov_frame_maps(
                        attn_probs=attn,                                # [B, H, P, N+P]
                        text_query_idx=first_gen_idx,                   # [B, 1]
                        frame_patch_indices=frame_patch_indices_batch,  # list-of-list, indicating frame patch indices for every sample
                        aggregate_heads="mean",
                        aggregate_queries="mean",
                        renorm_per_frame=False
                    )

                ## frame_maps: [B, F, 16, 16]
                nss_scores_batch = compute_normalized_scanpath_saliency(attention_maps=frame_maps, scanpath=human_gaze, attention_shape=(224, 224))
                nss_scores.append(nss_scores_batch)

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
        handle.remove()
        nss_scores = torch.cat(nss_scores, dim=0)
        nss_avg_val = nss_scores.mean().item()
        nss_std_val = nss_scores.std().item()
        results.append({'Avg. NSS': nss_avg_val, 'Std. NSS': nss_std_val})

    return results, nss_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate Video-LLaVA model on HumanEyeCue dataset.")
    parser.add_argument('--model_dir', type=str, default="", help='Path to Video-LLaVA model directory')
    parser.add_argument('--model_id',  type=str, default='LanguageBind/Video-LLaVA-7B-hf', help='Id/Name of Video-LLaVA model')
    parser.add_argument('--data_cfg_file', type=str, default="data/model_inputs/model_config.json", help='Path to data config file')
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'test', 'trainval', 'all'], default='test',
                        help='Dataset split to evaluate on: train, valid, test, trainval, or all (default: valid)')
    parser.add_argument('--video_sampling_method', type=str, default='uniform', help='Method for sampling video frames')
    parser.add_argument('--video_num_frames', type=int, default=8, help='Number of video frames to sample')
    parser.add_argument('--attention_type', type=str, default=None, choices=[None, "spatial", "temporal", "spatiotemporal"], help='Attention type to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./models/video_llava/results/', help='Path to save the results JSON file')
    parser.add_argument('--experiment_name', type=str, default='zeroshot_inference_test_set_attn_layer_0', help='Experiment name used in naming the output files')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoLlavaForConditionalGeneration.from_pretrained(args.model_dir + args.model_id).to(device)
    processor = AutoProcessor.from_pretrained(args.model_dir + args.model_id)
    processor.tokenizer.padding_side = "right"

    dataloader = get_humaneyecue_dataloader(humaneyecue_config_file=args.data_cfg_file,
                                         split=args.split, batch_size=args.batch_size,
                                         video_sampling_method=args.video_sampling_method, video_num_frames=args.video_num_frames,
                                         attention_type=args.attention_type,
                                         processor=processor)

    results, nss_scores = evaluate(model, processor, dataloader, device)

    # Ensure output directory exists
    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON file
    with open(os.path.join(args.output_dir, f"{args.experiment_name}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {os.path.join(args.output_dir, f'{args.experiment_name}.json')}")

    if nss_scores is not None:
        with open(os.path.join(args.output_dir, f"{args.experiment_name}_nss.pkl"), 'wb') as f:
            pickle.dump(nss_scores.tolist(), f)
        print(f"NSS scores saved to {os.path.join(args.output_dir, f'{args.experiment_name}_nss.pkl')}")

if __name__ == "__main__":
    main()
