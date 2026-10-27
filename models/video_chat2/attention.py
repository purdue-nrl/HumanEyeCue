"""Original HumanEyeCue gaze saliency and NSS mathematics, with release adapters."""

import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def restore_attention(attention, keep_indices=None, frames=8):
    """Mean heads/queries and restore removed patches to zero for NSS.

    The filtered reconstruction is a release convention, not a verified match
    to the unpublished Table 4 aggregation.
    """
    values = attention.detach().float().mean(dim=1).mean(dim=1)
    if values.shape[0] != 1:
        raise ValueError('NSS evaluator requires batch size one')
    if keep_indices is not None:
        indices = keep_indices.to(values.device)
        restored = values.new_zeros((1, frames * 196))
        restored[:, indices] = values
        values = restored
    return values.reshape(1, frames, 14, 14)

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
    orig_dtype = dtype

    # 1) Optional resize (differentiable)
    if attention_shape is not None:
        new_h, new_w = attention_shape
        att = attention_maps.view(B * T, 1, H, W)
        # att = F.interpolate(att, size=(new_h, new_w), mode="bilinear", align_corners=False)
        att = F.interpolate(att.to(torch.float32), size=(new_h, new_w), mode="bilinear", align_corners=False)
        att = att.view(B, T, new_h, new_w)
        H, W = new_h, new_w
    else:
        att = attention_maps

    att = torch.nan_to_num(att, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)

    # 2) Per-frame z-score normalization
    mean = att.mean(dim=(-1, -2), keepdim=True)
    std  = att.std(dim=(-1, -2), keepdim=True, unbiased=False).clamp_min(eps)
    att_norm = (att - mean) / std  # [B, T, H, W]
    att_norm = torch.nan_to_num(att_norm, nan=0.0, posinf=0.0, neginf=0.0).to(orig_dtype)

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

            # Convert to float32 for grid_sample as it doesn't support BFloat16
            frame_map_float32 = frame_map.float()
            grid_float32 = grid.float()

            sampled = F.grid_sample(
                frame_map_float32, grid_float32, mode="bilinear",
                align_corners=False, padding_mode="zeros"
            )                                                    # [1,1,N,1]
            vals = sampled.view(-1).to(dtype)                    # [N] - convert back to original dtype
            frame_scores.append(vals.mean())

        nss = torch.tensor(float("nan"), dtype=dtype, device=device) if len(frame_scores) == 0 \
              else torch.stack(frame_scores).mean()
        nss_per_sample.append(nss)

    return torch.stack(nss_per_sample)

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
            valid = ~(~np.isfinite(xs) | ~np.isfinite(ys) | ~np.isfinite(ws))
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

def get_human_saliency_maps(human_gaze):
    ## 1. estimate the standard deviation of the Gaussian fovea in pixels for videos
    stdev_gaussian_fovea_screen = calculate_pixels_for_angle(
        viewing_distance_cm = 55,
        visual_angle_deg = 2,
        screen_resolution = (1080, 1920),
        screen_diagonal_inches = 24)
    stdev_gaussian_fovea_videos = (stdev_gaussian_fovea_screen * 480) / 1920

    ## 2. create human saliency maps for the videos in the batch
    human_saliency_maps = create_human_saliency_maps_from_gaze(
        human_gaze=human_gaze.detach().cpu().numpy(),
        saliency_map_size=(320, 480),
        sigma_px=stdev_gaussian_fovea_videos) # shape: (batch_size, num_frames, saliency_map_height, saliency_map_width)
    human_saliency_maps = torch.tensor(human_saliency_maps).to(human_gaze.device)  # convert to tensor and move to the same device as the input batch

    ## 3. downsample human saliency maps to match the size of the video frames returned by preprocessor
    B, T, H, W = human_saliency_maps.shape
    target_height, target_width = 224, 224 ### CLIP
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
