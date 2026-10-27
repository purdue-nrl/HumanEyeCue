import os
import json
import torch
import pickle
import argparse
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, AutoProcessor
from humaneyecue_dataloader import get_humaneyecue_dataloader, human_gaze_to_tensor
from humaneyecue_helper_functions import compute_normalized_scanpath_saliency, calculate_pixels_for_angle, create_human_saliency_maps_from_gaze, adjust_inputs_based_on_reduced_video_tokens

## Reproducibility
np.random.seed(19)
torch.manual_seed(19)
torch.cuda.manual_seed_all(19)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_collection_settings = {
    "video_resolution":          [320, 480],
    "screen_resolution":         [1080, 1920],
    "screen_diagonal_inches":    24,
    "viewing_distance_cm":       55,
    "visual_angle_degrees":      2
}

def get_human_saliency_maps(processor, human_gaze):
    ## 1. estimate the standard deviation of the Gaussian fovea in pixels for videos
    stdev_gaussian_fovea_screen = calculate_pixels_for_angle(
        viewing_distance_cm = data_collection_settings['viewing_distance_cm'],
        visual_angle_deg = data_collection_settings['visual_angle_degrees'],
        screen_resolution = tuple(data_collection_settings['screen_resolution']),
        screen_diagonal_inches = data_collection_settings['screen_diagonal_inches'])
    stdev_gaussian_fovea_videos = (stdev_gaussian_fovea_screen * data_collection_settings['video_resolution'][1]) / data_collection_settings['screen_resolution'][1]

    ## 2. create human saliency maps for the videos in the batch
    human_saliency_maps = create_human_saliency_maps_from_gaze(
        human_gaze=human_gaze.detach().cpu().numpy(),
        saliency_map_size=tuple(data_collection_settings['video_resolution']),
        sigma_px=stdev_gaussian_fovea_videos) # shape: (batch_size, num_frames, saliency_map_height, saliency_map_width)
    human_saliency_maps = torch.tensor(human_saliency_maps).to(human_gaze.device)  # convert to tensor and move to the same device as the input batch

    ## 3. downsample human saliency maps to match the size of the video frames returned by preprocessor
    B, T, H, W = human_saliency_maps.shape
    target_height, target_width = processor.video_processor.crop_size['height'], processor.video_processor.crop_size['width']
    human_saliency_maps = human_saliency_maps.view(B * T, 1, H, W)  # reshape to (B*T, 1, H, W) for interpolation
    downsampled_human_saliency_maps = torch.nn.functional.interpolate(
        human_saliency_maps,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )
    downsampled_human_saliency_maps = downsampled_human_saliency_maps.view(B, T, target_height, target_width)  # reshape back to (B, T, target_height, target_width)
    # normalize per-frame to [0,1]
    _flat = downsampled_human_saliency_maps.view(B, T, -1)
    mins = _flat.amin(dim=2).view(B, T, 1, 1)
    maxs = _flat.amax(dim=2).view(B, T, 1, 1)
    denom = maxs - mins
    # avoid division by zero
    denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)
    normalized = (downsampled_human_saliency_maps - mins) / denom_safe
    # for constant frames (denom==0) set to zeros
    downsampled_human_saliency_maps = torch.where(denom == 0, torch.zeros_like(normalized), normalized)
    return downsampled_human_saliency_maps


def get_human_saliency_visual_tokens_mask(model, processor, human_gaze, threshold):
    human_saliency_maps = get_human_saliency_maps(processor, human_gaze)
    # Average saliency maps over patches
    ksize   = model.vision_tower.vision_model.embeddings.patch_embedding.kernel_size
    stride  = model.vision_tower.vision_model.embeddings.patch_embedding.stride
    averaged_human_saliency_maps = torch.nn.functional.avg_pool2d(human_saliency_maps, kernel_size=ksize, stride=stride, ceil_mode=False)
    # Threshold patches based on saliency values
    binary_saliency_maps = (averaged_human_saliency_maps >= threshold).float()
    return binary_saliency_maps


# @jleehun
def text_pov_frame_maps_with_gaze(
    attn_probs: torch.Tensor,      # [B, H, N, N], Softmaxed over last dim (keys). N is *pruned*
    text_query_idx: torch.LongTensor,# [B, T] or [B, 1]
    frame_patch_indices: list,      # length B list; each item is list of length F of LongTensor [N_f]
    aggregate_heads: str = "mean",  # average over heads: "mean" | "max" | None
    aggregate_queries: str = "mean", # average over text queries: "mean" | None
    renorm_per_frame: bool = True,    # renormalize frame sub-maps so each sums to 1
    human_attention_visual_tokens_mask: torch.Tensor=None, # [B, F, 16, 16], binary mask
):
    """
    Returns:
      If aggregate_heads="mean" and aggregate_queries="mean": [B, F, 16, 16]
      If aggregate_heads=None and aggregate_queries="mean":   [B, H, F, 16, 16]
      If aggregate_heads="mean" and aggregate_queries=None:   [B, T, F, 16, 16]
      If neither:                                             [B, H, T, F, 16, 16]
    """
    B, H, N, _ = attn_probs.shape
    # Gather queries
    # Make text_query_idx shape [B, T]
    if text_query_idx.dim() == 2:
        T = text_query_idx.shape[1]
    elif text_query_idx.dim() == 1:
        text_query_idx = text_query_idx.unsqueeze(1)
        T = 1
    else:
        raise ValueError("text_query_idx must be [B, T] or [B]")

    # [B, H, T, N]
    q = []
    for b in range(B):
        # text_query_idx[b] are indices *within the pruned sequence N*
        q.append(attn_probs[b, :, text_query_idx[b], :])  # [H, T, N]
    q = torch.stack(q, dim=0)  # [B, H, T, N]

    per_frame = []

    if human_attention_visual_tokens_mask is None:
        for b in range(B):
            Fslices = []
            for idx in frame_patch_indices[b]:
                Fslices.append(q[b].index_select(dim=-1, index=idx))  # [H, T, 256]
            per_frame.append(torch.stack(Fslices, dim=2))  # [H, T, F, 256]
            num_height_side=16
            num_width_side=16
    else:
        num_height_side = human_attention_visual_tokens_mask.shape[-2]  # 16
        num_width_side = human_attention_visual_tokens_mask.shape[-1]  # 16
        # --- START: MODIFICATION ---
        # 'q' is pruned [B, H, T, N_pruned_total]. We must scatter it back to [B, H, T, F, 256]

        # Get num_patches (256)
        num_patches = human_attention_visual_tokens_mask.shape[-1] * human_attention_visual_tokens_mask.shape[-2] # 16*16

        for b in range(B):
            num_frames = len(frame_patch_indices[b]) # Number of frames for this batch item

            # Get the mask for this batch item, flattened to [F, 256]
            mask_flat = human_attention_visual_tokens_mask[b].view(num_frames, num_patches) # [F, 256]

            Fslices = []
            for frame_idx in range(num_frames):
                # idx contains the *global* indices (in N) for the N_f tokens of this frame
                idx = frame_patch_indices[b][frame_idx] # Shape [N_f]

                # Select the pruned attention values for *this frame*
                # Shape: [H, T, N_f]
                pruned_attn_slice = q[b].index_select(dim=-1, index=idx)

                # Get the *local* patch indices (0-255) for this frame from the mask
                # These are the "scatter" locations
                # Shape: [N_f]
                frame_mask_indices = mask_flat[frame_idx].nonzero(as_tuple=True)[0]

                # --- This is the key check ---
                # The number of selected values from 'q' must match the number
                # of 'True' locations in the mask for this frame.
                if pruned_attn_slice.shape[-1] != frame_mask_indices.shape[0]:
                    raise ValueError(
                        f"Frame {frame_idx} in batch {b}: Mismatch in token count. "
                        f"attn_probs gave {pruned_attn_slice.shape[-1]} tokens, "
                        f"but human_mask has {frame_mask_indices.shape[0]} tokens."
                    )

                # Create the full-size [H, T, 256] zero tensor
                full_frame_attn = torch.zeros(
                    H, T, num_patches,
                    device=q.device,
                    dtype=q.dtype
                )

                # Scatter the pruned values into the full tensor
                # [H, T, 256] <--- [H, T, N_f]
                full_frame_attn[:, :, frame_mask_indices] = pruned_attn_slice

                # We now have a full [H, T, 256] tensor (sparse)
                Fslices.append(full_frame_attn)

            # Stack all frames: [H, T, F, 256]
            per_frame.append(torch.stack(Fslices, dim=2))

    A = torch.stack(per_frame, dim=0)  # [B, H, T, F, 256]
    # --- END: MODIFICATION ---


    if renorm_per_frame:
        # This will now correctly sum over the 256 patches (most are 0)
        # and normalize the non-zero values relative to each other.
        denom = A.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [B, H, T, F, 1]
        A = A / denom

    # reshape to grids: [B, H, T, F, 16, 16]
    A = A.view(A.shape[0], A.shape[1], A.shape[2], A.shape[3], num_height_side, num_width_side)

    # Aggregate queries
    if aggregate_queries == "mean":
        A = A.mean(dim=2)  # [B, H, F, 16, 16]
    elif aggregate_queries is None:
        pass
    else:
        raise ValueError("aggregate_queries must be 'mean' or None")

    # Aggregate heads
    if aggregate_heads == "mean":
        A = A.mean(dim=1)  # [B, F, 16, 16]
    elif aggregate_heads == "max":
        A, _ = A.max(dim=1)  # [B, F, 16, 16]
    elif aggregate_heads is None:
        pass
    else:
        raise ValueError("aggregate_heads must be 'mean' | 'max' | None")

    return A

# @jleehun
def infer_text_video_spans_with_gaze(
    input_ids: torch.LongTensor,          # [B, N]
    video_token_id: int,                  # value in input_ids corresponding to video tokens
    patches_per_frame: int = 16*16,       # 256 patches into which each frame is divided
    tokens_per_frame: int = 1 + 16*16,     # 257 (CLS + patches) for CLIP model
    human_attention_visual_tokens_mask: torch.Tensor=None, # [B, num_frames, 16, 16]
    has_cls_token: bool = True            # New Option: set False if no CLS token per frame
):
    """
    Returns per-sample dict with:
      - text_all_idx: LongTensor [T] of all text token indices (pre + post)
      - text_last_idx: LongTensor [1] of the last text token index
      - video_start, video_end: ints (start inclusive, end exclusive)
      - num_frames: int
      - frame_patch_indices: list of length F of LongTensor [N_patches_f] (variable length)
      - frame_cls_indices:   LongTensor [F] (the per-frame CLS query indices)

    Calculates # tokens per frame dynamically using human_attention_visual_tokens_mask.
    """
    B, N = input_ids.shape
    out = []

    video_mask = (input_ids == video_token_id)  # [B, N]
    cls_offset = 1 if has_cls_token else 0

    for b in range(B):
        vm = video_mask[b]                      # [N]
        vid_pos = torch.nonzero(vm, as_tuple=False).flatten()
        assert vid_pos.numel() > 0, "No video tokens found."

        video_start = int(vid_pos[0].item())
        video_end   = int(vid_pos[-1].item()) + 1  # exclusive
        video_len   = video_end - video_start

        # Moved 1
        # assert video_len % tokens_per_frame == 0, \
        #     f"Video span ({video_len}) is not a multiple of tokens_per_frame={tokens_per_frame}."
        # num_frames = video_len // tokens_per_frame

        # text tokens = everything outside [video_start:video_end)
        pre_text  = torch.arange(0, video_start, device=input_ids.device)
        post_text = torch.arange(video_end, N, device=input_ids.device)
        text_all_idx  = torch.cat([pre_text, post_text], dim=0)
        assert text_all_idx.numel() > 0, "No text tokens detected outside video span."
        text_last_idx = text_all_idx[-1:].clone()  # [1]

        frame_patch_indices = []
        frame_cls_indices   = []

        # Add conditional branch
        if human_attention_visual_tokens_mask is None:

            # Moved 1
            assert video_len % tokens_per_frame == 0, \
                f"Video span ({video_len}) is not a multiple of tokens_per_frame={tokens_per_frame}."
            num_frames = video_len // tokens_per_frame

            # Same as before
            # Build per-frame indices
            for f in range(num_frames):
                frame_start = video_start + f * tokens_per_frame
                cls_idx     = frame_start  # first token in the frame block
                # only patches (exclude CLS)
                patch_idx   = torch.arange(frame_start + 1, frame_start + 1 + patches_per_frame, device=input_ids.device)
                frame_patch_indices.append(patch_idx)      # [256]
                frame_cls_indices.append(cls_idx)

            frame_cls_indices = torch.tensor(frame_cls_indices, device=input_ids.device, dtype=torch.long)

            out.append(dict(
                text_all_idx=text_all_idx,
                text_last_idx=text_last_idx,
                video_start=video_start,
                video_end=video_end,
                num_frames=num_frames,
                frame_patch_indices=frame_patch_indices,
                frame_cls_indices=frame_cls_indices
            ))

        else:
            # --- Modification Start ---
            # 1. Get pruning info for this batch item
            # mask shape: [num_frames, 16, 16]
            mask_b = human_attention_visual_tokens_mask[b]
            num_frames = mask_b.shape[0]

            # 2. Count kept patches per frame
            # Flatten spatial dims: [num_frames, 256]
            flat_mask = mask_b.view(num_frames, -1)

            # Count non-zeros: [num_frames]
            kept_patches_per_frame = torch.count_nonzero(flat_mask, dim=1)

            # Total tokens per frame = (CLS) + kept patches
            tokens_per_frame_list = (cls_offset + kept_patches_per_frame).tolist() # List of ints

            # 3. Verify consistency
            expected_len = sum(tokens_per_frame_list)
            if video_len != expected_len:
                raise ValueError(
                    f"Batch {b}: input_ids has {video_len} video tokens, but "
                    f"human_attention_visual_tokens_mask implies {expected_len} tokens "
                    f"(sum of 1+patches over {num_frames} frames)."
                )

            # 4. Build dynamic indices
            current_offset = video_start

            for f in range(num_frames):
                # Number of patches kept for this specific frame
                n_patches = tokens_per_frame_list[f] - cls_offset

                # 1. Handle CLS
                if has_cls_token:
                    frame_cls_indices.append(current_offset)

                p_start = current_offset + cls_offset
                p_end   = p_start + n_patches

                # The subsequent tokens are the visual patches
                # Range: [current + (CLS) , current + (CLS) + n_patches)
                patch_idx = torch.arange(
                    p_start,
                    p_end,
                    device=input_ids.device
                )

                frame_patch_indices.append(patch_idx)

                # Advance the pointer
                current_offset += tokens_per_frame_list[f]
            # --- Modification End ---

            frame_cls_indices = torch.tensor(frame_cls_indices, device=input_ids.device, dtype=torch.long)

            out.append(dict(
                text_all_idx=text_all_idx,
                text_last_idx=text_last_idx,
                video_start=video_start,
                video_end=video_end,
                num_frames=num_frames,
                frame_patch_indices=frame_patch_indices, # List of tensors (variable sizes)
                frame_cls_indices=frame_cls_indices      # Tensor [F] # Empty tensor if has_cls_token=False
            ))

    return out  # list of length B


def evaluate(model, processor, dataloader, device, attention_query="all_text_tokens", attention_layer_idx=-1):
    model.eval()
    if attention_query in ["all_text_tokens"]:
        ## Attention weights are not computed explicitly if attn_implementation is "sdpa" or "flash"
        model.language_model.set_attn_implementation("eager")
        ## Setting up attention hook on decode (llama model for video-llava)
        handles = []
        def attn_weight_hook(module, input, output, layer_idx):
            attention_weights_llama[layer_idx].append(output[1])

        if attention_layer_idx == 'all':
            attention_weights_llama = {i: [] for i in range(len(model.language_model.layers))}
            for i, layer in enumerate(model.language_model.layers):
                handle = layer.self_attn.register_forward_hook(lambda module, input, output, layer_idx=i: attn_weight_hook(module, input, output, layer_idx))
                handles.append(handle)
        elif isinstance(attention_layer_idx, int):
            attention_weights_llama = {attention_layer_idx: []}
            handle = model.language_model.layers[attention_layer_idx].self_attn.register_forward_hook(lambda module, input, output, layer_idx=attention_layer_idx: attn_weight_hook(module, input, output, layer_idx))
            handles.append(handle)
        else:
            raise ValueError(f"Received unexpected value for attention_layer_idx = {attention_layer_idx}.\n")
    elif attention_query == 'pred_text_token':
        raise NotImplementedError()
    elif attention_query is None:
        pass
    else:
        raise ValueError
    results = []
    video_tokens_kept = []
    nss_scores = {}
    strict_acc = {question_no: {'correct': 0, 'total': 0} for question_no in [0, 1, 2, 3, 4, 6, 51, 52, 53, 54, 71, 72, 73, 74, 'all']}  # question_no: [correct, total]

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if attention_query:
                ## Reset attention buffer
                for layer_idx in attention_weights_llama:
                    attention_weights_llama[layer_idx].clear()

            ## Forward pass
            input_ids, attention_mask, pixel_values_videos, (vq_pairs, texts, gt_answers, human_gaze) = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]
            human_gaze_tensor = human_gaze_to_tensor(human_gaze, device=device)
            model.vision_tower.vision_model.clear_human_attention()
            human_attention_visual_tokens_mask = get_human_saliency_visual_tokens_mask(model, processor, human_gaze_tensor, threshold=0.5)
            human_attention_visual_tokens_mask_reshaped = human_attention_visual_tokens_mask.reshape(pixel_values_videos.shape[0]*pixel_values_videos.shape[1], -1)
            video_tokens_kept.append(torch.count_nonzero(human_attention_visual_tokens_mask_reshaped, dim=1).cpu())
            model.vision_tower.vision_model.store_human_attention(human_attention_visual_tokens_mask_reshaped)
            input_ids, attention_mask = adjust_inputs_based_on_reduced_video_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                human_attention_visual_tokens_mask=human_attention_visual_tokens_mask,
                video_token_id=model.config.video_token_id,
                pad_token_id=model.config.pad_token_id,
            )

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values_videos=pixel_values_videos, max_new_tokens=50, do_sample=False)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)

            if attention_query:
                ## Analyze model attention
                ## INFO 1: llava-next-video REMOVES class embedding at the beginning of the video tokens BEFORE being processed by the language model
                ## INFO 2: default configurations preprocess video frames into 336x336 pixels dimensions, process them using kernel_size=14 and stride=14 convolution within CLIP, AND rescale video feature dimensions by 2
                ##         As a result, by default, there are (336/14)/2 = 12x12 total video tokens processed by the language model.
                ## INFO 2: attention_weights would be in the format [batch_size, num_heads, seq_length, seq_length], where seq_length is the sequence length in terms of tokens (BOTH text and video). In here, attn_weights[0, :, 4, :] would represent attention paid by token 4 (index 2) onto every other token (index 3).
                ## NOTE 3: currently, attention helper functions do not account for padding tokens, which might appear if batch_size > 1

                # Infer spans (text/video) per sample from input_ids
                spans = infer_text_video_spans_with_gaze(input_ids=input_ids, video_token_id=model.config.video_token_id,
                                                         patches_per_frame=144, tokens_per_frame=144,
                                                         human_attention_visual_tokens_mask=human_attention_visual_tokens_mask, has_cls_token=False)
                # Convenience: per-sample frame patch indices
                frame_patch_indices_batch = [s["frame_patch_indices"] for s in spans]

                for layer_idx in attention_weights_llama:
                    # --- Case 1: all text tokens as queries ---
                    attn = attention_weights_llama[layer_idx][0]            # [B, H, N, N]
                    B = attn.size(0)
                    frame_maps_list = []
                    for b in range(B):
                        text_idx_b = spans[b]["text_all_idx"].unsqueeze(0)  # <-- every sample will have different prompt, and hence different order of text tokens
                        maps_b = text_pov_frame_maps_with_gaze(
                            attn_probs=attn[b:b+1],                                 # [1, H, N, N]
                            text_query_idx=text_idx_b,                              # [1, T_b]
                            frame_patch_indices=[frame_patch_indices_batch[b]],     # list-of-list, indicating frame patch indices for this sample
                            aggregate_heads="mean",
                            aggregate_queries="mean",
                            renorm_per_frame=False,
                            human_attention_visual_tokens_mask=human_attention_visual_tokens_mask,
                        )
                        frame_maps_list.append(maps_b)  # list of length B; each item is [1, F, 16, 16]
                    frame_maps = torch.cat(frame_maps_list, dim=0)  # [B, F, 16, 16]
                    ## frame_maps: [B, F, 16, 16]
                    nss_scores_batch = compute_normalized_scanpath_saliency(attention_maps=frame_maps, scanpath=human_gaze, attention_shape=(336, 336))
                    if layer_idx not in nss_scores:
                        nss_scores[layer_idx] = []
                    nss_scores[layer_idx].append(nss_scores_batch)

            ## Record results
            for vq, qp, a, p in zip(vq_pairs, texts, gt_answers, preds):
                results.append({'video_no': vq[0], 'question_no': vq[1], 'prompt': qp, 'ground_truth_answer': a, 'predicted_answer': p})
                # Record strict accuracy, i.e., the model has to match the gt answer exactly.
                model_answer = p.split("ASSISTANT: Answer: ")[1]
                model_answer = model_answer.lstrip()
                if len(model_answer) > 0 and model_answer[0] == a:
                    strict_acc[vq[1]]['correct'] += 1
                    strict_acc['all']['correct'] += 1
                strict_acc[vq[1]]['total'] += 1
                strict_acc['all']['total'] += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches")

    results.append(strict_acc)
    video_tokens_kept = torch.stack(video_tokens_kept, dim=0) # [num_samples, num_frames]
    avg_tokens_kept = video_tokens_kept.float().mean().item()
    std_tokens_kept = video_tokens_kept.float().std().item()
    results.append({f'Avg. video tokens kept per frame': avg_tokens_kept, f'Std. video tokens kept per frame': std_tokens_kept})
    if attention_query:
        for handle in handles:
            handle.remove()
        for layer_idx in nss_scores:
            nss_scores[layer_idx] = torch.cat(nss_scores[layer_idx], dim=0)
            nss_avg_val = nss_scores[layer_idx].mean().item()
            nss_std_val = nss_scores[layer_idx].std().item()
            results.append({f'Avg. NSS @ {layer_idx}': nss_avg_val, f'Std. NSS @ {layer_idx}': nss_std_val})

    return results, video_tokens_kept, nss_scores

def int_or_all(value):
    if value.lower() == 'all':
        return 'all'
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("attention_layer must be an integer or 'all'.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Video-LLaVA model on HumanEyeCue dataset.")
    parser.add_argument('--model_dir', type=str, default="", help='Path to LLaVA-NeXT-Video model directory')
    parser.add_argument('--model_id',  type=str, default='llava-hf/LLaVA-NeXT-Video-7B-hf', help='Id/Name of LLaVA-NeXT-Video model')
    parser.add_argument('--finetuned_model_path', type=str, default="./checkpoints/REPLACE_WITH_CHECKPOINT", help='Path to a fine-tuned model checkpoint (if any)')
    parser.add_argument('--data_cfg_file', type=str, default="data/model_inputs/model_config.json", help='Path to data config file')
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'test', 'trainval', 'all'], default='test',
                        help='Dataset split to evaluate on: train, valid, test, trainval, or all (default: valid)')
    parser.add_argument('--video_sampling_method', type=str, default='uniform', help='Method for sampling video frames')
    parser.add_argument('--video_num_frames', type=int, default=16, help='Number of video frames to sample')
    parser.add_argument('--attention_type', type=str, default='spatial', choices=[None, "spatial", "temporal", "spatiotemporal"], help='Attention type to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./models/llava_next_video/results/', help='Path to save the results JSON file')
    parser.add_argument('--experiment_name', type=str, default='inference_finetuned_on_vq_and_gaze_test_set', help='Experiment name used in naming the output files')
    parser.add_argument('--attention_query', type=str, default='all_text_tokens', choices=['all_text_tokens', 'None'], help='Which text tokens to use as queries for attention analysis')
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

    results, video_tokens_kept, nss_scores = evaluate(model, processor, dataloader, device, attention_query=args.attention_query if args.attention_query != 'None' else None, attention_layer_idx=args.attention_layer)

    # Ensure output directory exists
    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON file
    with open(os.path.join(args.output_dir, f"{args.experiment_name}_attn_layer_{args.attention_layer}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {os.path.join(args.output_dir, f'{args.experiment_name}_attn_layer_{args.attention_layer}.json')}")

    with open(os.path.join(args.output_dir, f"{args.experiment_name}_attn_layer_{args.attention_layer}_video_tokens_kept.pkl"), 'wb') as f:
        pickle.dump(video_tokens_kept, f)
    print(f"Video tokens kept data saved to {os.path.join(args.output_dir, f'{args.experiment_name}_attn_layer_{args.attention_layer}_video_tokens_kept.pkl')}")

    if nss_scores is not None:
        with open(os.path.join(args.output_dir, f"{args.experiment_name}_attn_layer_{args.attention_layer}_nss.pkl"), 'wb') as f:
            pickle.dump(nss_scores, f)
        print(f"NSS scores saved to {os.path.join(args.output_dir, f'{args.experiment_name}_attn_layer_{args.attention_layer}_nss.pkl')}")

if __name__ == "__main__":
    main()
