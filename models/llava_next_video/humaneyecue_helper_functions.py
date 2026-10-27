import torch
import torch.nn.functional as F
from typing import Dict, Iterable, Tuple, Union

# [B, H, N, N] -> [B, H, T (text tokens), F (num_frames), 12, 12]

def infer_text_video_spans(
    input_ids: torch.LongTensor,          # [B, N]
    video_token_id: int,                  # value in input_ids corresponding to video tokens
    patches_per_frame: int = 12*12,       # 144 patches into which each frame is divided
    tokens_per_frame: int = 0 + 12*12     # 144 (no CLS + patches) passed from CLIP to language decoder model for LLaVA-NeXT-Video model
):
    # FIXME: currently does not handle padding tokens
    """
    Returns per-sample dict with:
      - text_all_idx: LongTensor [T] of all text token indices (pre + post)
      - text_last_idx: LongTensor [1] of the last text token index
      - video_start, video_end: ints (start inclusive, end exclusive)
      - frame_patch_indices: list of length F of LongTensor [patches_per_frame] (patch keys only)
      - frame_cls_indices:   LongTensor [F] (the per-frame CLS query indices)
    Auto-infers #frames from contiguous video span length.
    """
    B, N = input_ids.shape
    out = []

    video_mask = (input_ids == video_token_id)  # [B, N]

    for b in range(B):
        vm = video_mask[b]                      # [N]
        vid_pos = torch.nonzero(vm, as_tuple=False).flatten()
        assert vid_pos.numel() > 0, "No video tokens found."

        video_start = int(vid_pos[0].item())
        video_end   = int(vid_pos[-1].item()) + 1  # exclusive
        video_len   = video_end - video_start

        assert video_len % tokens_per_frame == 0, \
            f"Video span ({video_len}) is not a multiple of tokens_per_frame={tokens_per_frame}."
        num_frames = video_len // tokens_per_frame

        # text tokens = everything outside [video_start:video_end)
        pre_text  = torch.arange(0, video_start, device=input_ids.device)
        post_text = torch.arange(video_end, N, device=input_ids.device)
        text_all_idx  = torch.cat([pre_text, post_text], dim=0)
        assert text_all_idx.numel() > 0, "No text tokens detected outside video span."
        text_last_idx = text_all_idx[-1:].clone()  # [1]

        # Build per-frame indices
        if tokens_per_frame - patches_per_frame == 1: # a single CLS token per frame
            frame_patch_indices = []
            frame_cls_indices   = []
            for f in range(num_frames):
                frame_start = video_start + f * tokens_per_frame
                cls_idx     = frame_start  # first token in the frame block
                # only patches (exclude CLS)
                patch_idx   = torch.arange(frame_start + 1, frame_start + 1 + patches_per_frame, device=input_ids.device)
                frame_patch_indices.append(patch_idx)      # list of length num_frames with each element being tensor of shape: [patches_per_frame]
                frame_cls_indices.append(cls_idx)
            frame_cls_indices = torch.tensor(frame_cls_indices, device=input_ids.device, dtype=torch.long)

        elif tokens_per_frame == patches_per_frame: # no CLS token per frame
            frame_patch_indices = []
            frame_cls_indices   = None
            for f in range(num_frames):
                frame_start = video_start + f * tokens_per_frame
                # only patches
                patch_idx   = torch.arange(frame_start, frame_start + patches_per_frame, device=input_ids.device)
                frame_patch_indices.append(patch_idx)       # list of length num_frames with each element being tensor of shape: [patches_per_frame]

        else:
            raise NotImplementedError("infer_text_video_spans is only implemented either for a single CLS token or no CLS token per frame.")

        out.append(dict(
            text_all_idx=text_all_idx,
            text_last_idx=text_last_idx,
            video_start=video_start,
            video_end=video_end,
            num_frames=num_frames,
            frame_patch_indices=frame_patch_indices,
            frame_cls_indices=frame_cls_indices
        ))

    return out  # list of length B



def text_pov_frame_maps(
    attn_probs: torch.Tensor,                       # [B, H, N, N], Softmaxed over last dim (keys)
    text_query_idx: torch.LongTensor,               # [B, T] or [B, 1]
    frame_patch_indices: list,                      # length B list; each item is list of length F of LongTensor [num_visual_patches_per_frame]
    visual_attn_shape: Tuple[int, int] = (12, 12),  # (H_a, W_a) spatial shape of visual attention maps per frame, s.t. num_visual_patches_per_frame = visual_attn_shape[0]*visual_attn_shape[1]
    aggregate_heads: str = "mean",                  # average over heads: "mean" | "max" | None
    aggregate_queries: str = "mean",                # average over text queries: "mean" | None
    renorm_per_frame: bool = True                   # renormalize frame sub-maps so each sums to 1
):
    """
    Returns:
      If aggregate_heads="mean" and aggregate_queries="mean": [B, F, visual_attn_shape[0], visual_attn_shape[1]]
      If aggregate_heads=None and aggregate_queries="mean":   [B, H, F, visual_attn_shape[0], visual_attn_shape[1]]
      If aggregate_heads="mean" and aggregate_queries=None:   [B, T, F, visual_attn_shape[0], visual_attn_shape[1]]
      If neither:                                             [B, H, T, F, visual_attn_shape[0], visual_attn_shape[1]]
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
        q.append(attn_probs[b, :, text_query_idx[b], :])  # [H, T, N]
    q = torch.stack(q, dim=0)  # [B, H, T, N]

    # Gather per frame -> [B, H, T, F, num_visual_patches_per_frame]
    per_frame = []
    for b in range(B):
        Fslices = []
        for idx in frame_patch_indices[b]:
            Fslices.append(q[b].index_select(dim=-1, index=idx))  # [H, T, num_visual_patches_per_frame]
        per_frame.append(torch.stack(Fslices, dim=2))  # [H, T, F, num_visual_patches_per_frame]
    A = torch.stack(per_frame, dim=0)  # [B, H, T, F, num_visual_patches_per_frame]

    if renorm_per_frame:
        denom = A.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [B, H, T, F, 1]
        A = A / denom

    # reshape to grids: [B, H, T, F, visual_attn_shape[0], visual_attn_shape[1]]
    A = A.view(A.shape[0], A.shape[1], A.shape[2], A.shape[3], visual_attn_shape[0], visual_attn_shape[1])

    # Aggregate queries
    if aggregate_queries == "mean":
        A = A.mean(dim=2)  # [B, H, F, visual_attn_shape[0], visual_attn_shape[1]]
    elif aggregate_queries is None:
        pass
    else:
        raise ValueError("aggregate_queries must be 'mean' or None")

    # Aggregate heads
    if aggregate_heads == "mean":
        A = A.mean(dim=1)  # [B, F, visual_attn_shape[0], visual_attn_shape[1]]
    elif aggregate_heads == "max":
        A, _ = A.max(dim=1)  # [B, F, visual_attn_shape[0], visual_attn_shape[1]]
    elif aggregate_heads is None:
        pass
    else:
        raise ValueError("aggregate_heads must be 'mean' | 'max' | None")

    return A



### Function to estimate normalized scanpath saliency (NSS) scores based on attention_maps from the model and human scanpaths

import torch
import torch.nn.functional as F

def compute_normalized_scanpath_saliency(attention_maps, scanpath, attention_shape=None, eps=1e-6):
    """
    Compute NSS per sample.

    Args:
        attention_maps: torch.Tensor [B, T, H, W]
        scanpath: list length B; each element is a list length T;
                  each inner list contains N (x, y) tuples in [0,1).
        attention_shape: optional (new_h, new_w). If provided, resize maps first.

    Returns:
        torch.Tensor [B] with NSS averaged over frames that have ≥1 valid fixation;
        NaN for samples with no valid frames.
    """
    assert attention_maps.dim() == 4, "attention_maps must be [B, T, H, W]"
    B, T, H, W = attention_maps.shape
    device, dtype = attention_maps.device, attention_maps.dtype

    # 1) Optional resize (differentiable)
    if attention_shape is not None:
        new_h, new_w = attention_shape
        att = attention_maps.view(B * T, 1, H, W)
        att = F.interpolate(att, size=(new_h, new_w), mode="bilinear", align_corners=False)
        att = att.view(B, T, new_h, new_w)
        H, W = new_h, new_w
    else:
        att = attention_maps

    # 2) Per-frame z-score normalization
    mean = att.mean(dim=(-1, -2), keepdim=True)
    std  = att.std(dim=(-1, -2), keepdim=True, unbiased=False).clamp_min(eps)
    att_norm = (att - mean) / std  # [B, T, H, W]

    # 3–4) Sample fixations → mean per frame → mean across frames
    nss_per_sample = []
    for i in range(B):
        frame_scores = []
        sp_i = scanpath[i] if i < len(scanpath) else [[] for _ in range(T)]
        for t in range(T):
            pts = sp_i[t] if t < len(sp_i) else []
            if not pts:
                continue

            # keep only fixations with 0 <= x,y < 1
            valid = [(float(x), float(y)) for (x, y) in pts if (0.0 <= x < 1.0) and (0.0 <= y < 1.0)]
            if len(valid) == 0:
                continue

            # build grid for bilinear sampling (expects [-1, 1] coords)
            xy = torch.tensor(valid, dtype=dtype, device=device)  # [N, 2] in [0,1)
            grid = xy * 2.0 - 1.0                                # -> [-1,1)
            grid = grid.view(1, -1, 1, 2)                        # [1, N, 1, 2]

            frame_map = att_norm[i, t].view(1, 1, H, W)          # [1,1,H,W]
            sampled = F.grid_sample(
                frame_map, grid, mode="bilinear",
                align_corners=False, padding_mode="zeros"
            )                                                    # [1,1,N,1]
            vals = sampled.view(-1)                              # [N]
            frame_scores.append(vals.mean())

        nss = torch.tensor(float("nan"), dtype=dtype, device=device) if len(frame_scores) == 0 \
              else torch.stack(frame_scores).mean()
        nss_per_sample.append(nss)

    return torch.stack(nss_per_sample)  # [B]


### Function to convert human gaze data into saliency maps
import numpy as np
import cv2

def calculate_pixels_for_angle(viewing_distance_cm = 55, visual_angle_deg = 2, screen_resolution = (1080, 1920),  screen_width_cm = None, screen_diagonal_inches = None):
    # Each fixation would be converted into a 2D Gaussian with σ = 2° visual angle (accounting for ~1° foveal radius + ~1° tracker noise).
    if (screen_width_cm is None) and (screen_diagonal_inches is None):
        raise ValueError("Please, specify either screen diagonal in inches or screen width in centemiers!")

    if screen_width_cm is None:
        screen_diagonal_cm = screen_diagonal_inches * 2.54
        screen_diagonal_px = np.hypot(screen_resolution[0], screen_resolution[1])
        pixel_pitch_cm = screen_diagonal_cm / screen_diagonal_px
    else:
        pixel_pitch_cm = float(screen_width_cm) / screen_resolution[1]
    # Calculate pixels per angle
    ppa = 2 * viewing_distance_cm * np.tan(np.radians(visual_angle_deg/2)) / pixel_pitch_cm
    return ppa


def create_human_saliency_maps_from_gaze(
    human_gaze: np.ndarray,
    saliency_map_size=(320, 480),           # (H_v, W_v)
    sigma_px=17.345,                        # float or (sigma_y, sigma_x) in *video pixels*
    coords_space: str = "normalized",       # "normalized" in [0,1), or "pixels"
    source_resolution: tuple | None = None, # (H_src, W_src) required if coords_space="pixels"
    weights: np.ndarray | None = None,      # optional weights same shape as [B,T,R]
    clamp_oob: bool = False,                # if True, clamp oob to borders; else discard (oob = out-of-bounds)
) -> np.ndarray:
    """
    Convert gaze points into per-frame human saliency maps by placing impulses,
    Gaussian-blurring, then normalizing to a probability distribution.

    Inputs
    ------
    human_gaze: np.ndarray
        Shape [B, T, R, 2] with (x, y) per recording. Depending on coords_space:
          - "normalized": x in [0,1), y in [0,1)
          - "pixels": x in [0, W_src), y in [0, H_src) where source_resolution=(H_src, W_src)
    saliency_map_size: tuple
        (H_v, W_v) of the output maps, typically your video size.
    sigma_px: float or (sigma_y, sigma_x)
        Gaussian standard deviation(s) in *video pixels*. If float, same σ for x and y.
        This should already reflect your chosen visual-angle model (e.g., σ = 2° converted to pixels).
    coords_space: str
        "normalized" or "pixels".
    source_resolution: tuple | None
        Required when coords_space="pixels". The source lattice the gaze lives in.
    weights: np.ndarray | None
        Optional weights per recording (e.g., fixation duration or confidence). Shape [B, T, R].
        If None, each valid point has weight 1.
    clamp_oob: bool
        If False (default), points outside the valid region are ignored. If True, they are clamped to border.

    Returns
    -------
    saliency_bthw: np.ndarray
        Float32 array of shape [B, T, H_v, W_v], each map sums to 1 if at least one valid point was present,
        otherwise the map is all zeros.
    """
    assert coords_space in ("normalized", "pixels")
    H_v, W_v = int(saliency_map_size[0]), int(saliency_map_size[1])

    if isinstance(sigma_px, (list, tuple, np.ndarray)):
        sigma_y, sigma_x = float(sigma_px[0]), float(sigma_px[1])
    else:
        sigma_y = sigma_x = float(sigma_px)

    if coords_space == "pixels":
        if source_resolution is None:
            raise ValueError("source_resolution=(H_src, W_src) must be provided when coords_space='pixels'.")
        H_src, W_src = int(source_resolution[0]), int(source_resolution[1])

    B, T, R, two = human_gaze.shape
    assert two == 2, "Last dim must be (x, y)."

    # Prepare weights
    if weights is None:
        weights = np.ones((B, T, R), dtype=np.float32)
    else:
        assert weights.shape == (B, T, R), "weights must have shape (B, T, R)."

    saliency = np.zeros((B, T, H_v, W_v), dtype=np.float32)

    for b in range(B):
        for t in range(T):
            # Impulse image for this frame
            frame_map = np.zeros((H_v, W_v), dtype=np.float32)

            xs = human_gaze[b, t, :, 0]
            ys = human_gaze[b, t, :, 1]
            ws = weights[b, t, :].astype(np.float32)

            # Filter out NaNs
            valid = ~(np.isnan(xs) | np.isnan(ys) | np.isnan(ws))
            if not np.any(valid):
                # leave zeros; no valid points
                saliency[b, t] = frame_map
                continue

            xs = xs[valid]
            ys = ys[valid]
            ws = ws[valid]

            # Map to video pixel indices
            if coords_space == "normalized":
                # x in [0,1), y in [0,1)
                xi = np.floor(xs * W_v).astype(np.int64)
                yi = np.floor(ys * H_v).astype(np.int64)
            else:
                # pixels on source lattice → scale to video lattice
                xi = np.floor(xs * (W_v / W_src)).astype(np.int64)
                yi = np.floor(ys * (H_v / H_src)).astype(np.int64)

            if clamp_oob:
                xi = np.clip(xi, 0, W_v - 1)
                yi = np.clip(yi, 0, H_v - 1)
                inb = np.ones_like(xi, dtype=bool)
            else:
                inb = (xi >= 0) & (xi < W_v) & (yi >= 0) & (yi < H_v)

            if not np.any(inb):
                saliency[b, t] = frame_map
                continue

            xi = xi[inb]
            yi = yi[inb]
            ws = ws[inb]

            # Average over recordings: match your previous approach
            wsum = ws.sum()
            if wsum <= 0:
                saliency[b, t] = frame_map
                continue
            ws = ws / wsum

            # Splat impulses with weights
            # Use flat indexing for speed
            inds = yi * W_v + xi
            np.add.at(frame_map.ravel(), inds, ws)

            # Gaussian blur with given sigma(s). Kernel size (0,0) lets OpenCV derive from sigma.
            blurred = cv2.GaussianBlur(frame_map, ksize=(0, 0), sigmaX=sigma_x, sigmaY=sigma_y)

            s = float(blurred.sum())
            if s > 0:
                blurred /= s
            # else keep zeros

            saliency[b, t] = blurred

    return saliency


@torch.no_grad()
def downsample_human_saliency_maps(
    human_saliency_maps: torch.Tensor,                  # [B, T, H, W], float
    target_size: tuple[int, int] = (16, 16),            # (Ha, Wa)
    frame_size: tuple[int, int] | None = (224, 224),    # default square frame
    use_interpolation: bool = True,                     # default: do pre-resize
    nondivisible_strategy: str = "resize",              # "resize" | "adaptive" | "pad"
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Downsample saliency maps by (optional) antialiased resize to frame_size,
    then non-overlapping bin integration to target_size (via avg-pool for equal bins),
    then renormalize to sum=1 per map.
    """
    assert human_saliency_maps.ndim == 4
    B, T, H, W = human_saliency_maps.shape
    Ha, Wa = target_size

    x = human_saliency_maps

    # Ensure non-negative & finite (defensive)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.clamp(x, min=0.0)

    # Normalize each map to sum=1
    x_flat = x.view(B * T, -1)
    sums = x_flat.sum(dim=1, keepdim=True)                 # [B*T, 1]
    x_flat = x_flat / sums.clamp_min(eps)                  # zero maps stay zero
    x = x_flat.view(B, T, H, W)

    # Optional interpolation to "frame_size" (e.g., 224x224)
    if use_interpolation and frame_size is not None and (frame_size[0] != H or frame_size[1] != W):
        x = x.view(B * T, 1, H, W)
        x = F.interpolate(x, size=frame_size, mode="bilinear", align_corners=False, antialias=True)
        x = x.view(B, T, frame_size[0], frame_size[1])
        H, W = frame_size

        # re-normalize after interpolation
        x_flat = x.view(B * T, -1)
        sums = x_flat.sum(dim=1, keepdim=True)
        x_flat = x_flat / sums.clamp_min(eps)
        x = x_flat.view(B, T, H, W)

    # --- Sum-pooling to (Ha, Wa) ---
    x = x.view(B * T, 1, H, W)

    if (H % Ha == 0) and (W % Wa == 0): # Perfectly divisible: equal-area non-overlapping bins
        kh, kw = H // Ha, W // Wa
        # avg-pool over equal bins; since all bins same area, avg ∝ sum; renorm later
        x = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw), ceil_mode=False)
    else:
        if nondivisible_strategy == "resize": # Find nearest size divisible by (Ha, Wa)
            Hd = ((H + Ha - 1) // Ha) * Ha
            Wd = ((W + Wa - 1) // Wa) * Wa
            x = F.interpolate(x, size=(Hd, Wd), mode="bilinear", align_corners=False, antialias=True)
            kh, kw = Hd // Ha, Wd // Wa
            x = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw), ceil_mode=False)
        elif nondivisible_strategy == "adaptive": # Unequal bins; renorm below fixes mass
            x = F.adaptive_avg_pool2d(x, output_size=(Ha, Wa))
        elif nondivisible_strategy == "pad": # Zero-pad to next divisible size, then equal-area avg-pool.
            Hd = ((H + Ha - 1) // Ha) * Ha
            Wd = ((W + Wa - 1) // Wa) * Wa
            pad_h, pad_w = Hd - H, Wd - W
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            kh, kw = Hd // Ha, Wd // Wa
            x = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw), ceil_mode=False)
        else:
            raise ValueError(f"Unknown nondivisible_strategy='{nondivisible_strategy}'")

    # Final renormalization to make each map sum to 1
    x = x.view(B, T, Ha, Wa)
    x_flat = x.view(B * T, -1)
    sums = x_flat.sum(dim=1, keepdim=True)
    x_flat = x_flat / sums.clamp_min(eps)
    x = x_flat.view(B, T, Ha, Wa)
    return x


def adjust_inputs_based_on_reduced_video_tokens(
    input_ids: torch.Tensor,              # [B, T]
    attention_mask: torch.Tensor,         # [B, T]
    human_attention_visual_tokens_mask: torch.Tensor,  # [B, num_frames, 24, 24] (0/1 or bool)
    video_token_id: int,
    pad_token_id: int,
):
    """
    For each sample:
    - Detect the contiguous [video_start:video_end) segment where input_ids == video_token_id.
    - For each frame, keep 1 + (#selected patches) tokens from that frame's 257-token block.
    - Trim the video segment accordingly.
    Returns padded tensors: new_input_ids [B, T_max], new_attention_mask [B, T_max]
    """
    B, T = input_ids.shape
    assert B == 1, "Currently only batch size 1 is supported."
    _, num_frames, H, W = human_attention_visual_tokens_mask.shape
    patches_per_frame = H * W  # 24x24=576 for LLaVANeXTVideo

    new_ids_list = []
    new_mask_list = []

    for b in range(B):
        ids_b   = input_ids[b]          # [T]
        mask_b  = attention_mask[b]     # [T]
        frame_sel = human_attention_visual_tokens_mask[b]  # [num_frames, 16, 16]

        # Locate the video segment (assumes one contiguous chunk of video_token_id)
        vid_pos = torch.nonzero(ids_b == video_token_id, as_tuple=False).squeeze(-1)
        if vid_pos.numel() == 0:
            # No video tokens? Keep as is.
            new_ids_list.append(ids_b)
            new_mask_list.append(mask_b)
            continue

        video_start = vid_pos.min().item()
        video_end   = vid_pos.max().item() + 1  # exclusive

        # Optional sanity: the length of video tokens in input_ids should be at least the number of tokens selected based on human attention
        min_expected    = frame_sel.count_nonzero().item()
        actual          = video_end - video_start
        if actual < min_expected:
            raise ValueError(f"[b={b}] Expected at least {min_expected} video tokens, but found only {actual} in input_ids.")

        # Count selected patches per frame
        # frame_keep = 0 (leading token) + number of selected patches <---- LLaVA-NeXT-Video does not keep CLS token from CLIP
        # (treat any nonzero as selected)
        selected_counts = torch.count_nonzero(frame_sel.view(num_frames, -1), dim=1)  # [num_frames]
        per_frame_keeps = (0 + selected_counts).tolist()  # python ints

        # Build the trimmed video segment: just repeat the same video_token_id
        # We keep the *first* K tokens of each frame-block. Since every token in the block
        # is the same id (video_token_id), order within the block doesn't matter.
        keep_ids = []
        for f_idx in range(num_frames):
            k = per_frame_keeps[f_idx]
            keep_ids.append(torch.full((k,), video_token_id, dtype=ids_b.dtype, device=ids_b.device))
        kept_video = torch.cat(keep_ids, dim=0)  # [sum_k]

        # Rebuild input_ids: [prefix prompt ...] + kept_video + [... suffix prompt]
        new_ids = torch.cat([ids_b[:video_start], kept_video, ids_b[video_end:]], dim=0)

        # Rebuild attention_mask: keep 1s for kept_video length
        kept_len = kept_video.size(0)
        new_mask = torch.cat([mask_b[:video_start],
                            torch.ones(kept_len, dtype=mask_b.dtype, device=mask_b.device),
                            mask_b[video_end:]], dim=0)

        new_ids_list.append(new_ids)
        new_mask_list.append(new_mask)

    # # Pad back to a batch (left as-is order-wise)
    # new_input_ids = torch.nn.utils.rnn.pad_sequence(new_ids_list, batch_first=True, padding_value=pad_token_id)  # [B, T_max]
    # new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_mask_list, batch_first=True, padding_value=0)       # [B, T_max]
    new_input_ids = torch.stack(new_ids_list, dim=0)
    new_attention_mask = torch.stack(new_mask_list, dim=0)

    return new_input_ids, new_attention_mask




### The loss function between model attention maps and human saliency maps
def compute_attention_loss(
    model_attention: Dict[int, torch.Tensor],                   # {layer_idx: [B, T, H, W]}
    human_attention: torch.Tensor,                              # [B, T, H, W]
    attention_lambda: Union[float, Iterable[float]] = 1.0,
    normalize_model: bool = False,                              # softmax the model per frame over spatial tokens
    temperature: float = 1.0,                                   # softmax temperature if normalize_model=True
    reduction: str = "mean",                                    # "mean" | "sum"
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Compute KL divergence between human attention maps (ground truth) and model attention maps,
    i.e. KL(G || Ahat) per (B,T), averaged (or summed) over valid frames.
    - model_attention: dict {layer_idx: [B, T, H, W]}, one or many layers
    - human_attention:  [B, T, H, W], ground-truth saliency/attention maps
    - attention_lambda: float or list of floats; per-layer weights in the same order as sorted(layer_idx)
    - normalize_model:  if True, applies softmax over spatial positions (per frame) to get Â
                        (if False, normalizes by sum like human maps)
    Returns:
      total_loss (scalar tensor), per_layer_losses (dict layer_idx -> scalar tensor)
    """
    assert isinstance(model_attention, dict) and len(model_attention) >= 1, "model_attention must be a non-empty dict"
    B, T, Hh, Wh = human_attention.shape

    # --- Prepare human attention as probability over spatial grid, per (B,T) ---
    G = human_attention.view(B * T, -1)                 # Flatten spatial: [B*T, H*W]
    G_sum = G.sum(dim=1, keepdim=True)                  # [B*T,1]
    valid_mask = (G_sum > 0).squeeze(1).float()         # [B*T], 1 for valid (i.e., have nonzero human mass), 0 for invalid
    G = G / G_sum.clamp_min(eps)                        # zero maps stay zero after divide
    logG = torch.log(G.clamp_min(eps))


    # layer weights
    layer_keys = sorted(model_attention.keys())
    if isinstance(attention_lambda, (float, int)):
        lambdas = [float(attention_lambda)] * len(layer_keys)
    else:
        lambdas = list(attention_lambda)
        if len(lambdas) != len(layer_keys):
            raise ValueError(f"attention_lambda length ({len(lambdas)}) != number of layers ({len(layer_keys)})")

    per_layer: Dict[int, torch.Tensor] = {}
    layer_terms = []

    for lk, lam in zip(layer_keys, lambdas):
        A = model_attention[lk]
        if A.shape[:2] != (B, T) or A.shape[2:] != (Hh, Wh):
            raise ValueError(f"Layer {lk} shape {A.shape} incompatible with human {human_attention.shape}")

        A_flat = A.view(B * T, -1)                      # Flatten spatial: [B*T, H*W]
        if normalize_model:                             # Softmax over spatial tokens per frame
            logits = A_flat / float(temperature)
            Ahat = F.softmax(logits, dim=1)
        else:                                           # Normalize like human attention (divide-by-sum)
            A_sum = A_flat.sum(dim=1, keepdim=True)
            Ahat = A_flat / A_sum.clamp_min(eps)

        # KL(G || Ahat) per sample: sum_i G_i * (log G_i - log Ahat_i)
        logA = torch.log(Ahat.clamp_min(eps))
        kl_per = (G * (logG - logA)).sum(dim=1)         # [B*T]

        # Mask invalid (no human mass) frames without branching
        masked_sum = (kl_per * valid_mask).sum()

        if reduction == "mean":
            denom = valid_mask.sum().clamp_min(1.0)          # avoids divide-by-zero; yields 0 if all invalid
            layer_loss = masked_sum / denom
        elif reduction == "sum":
            layer_loss = masked_sum
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

        per_layer[lk] = layer_loss.detach()
        layer_terms.append(lam * layer_loss)

    total_loss = torch.stack(layer_terms).sum()
    return total_loss, per_layer
