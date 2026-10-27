import torch
import numpy as np
from humaneyecue_dataset import HumanEyeCue
from transformers import AutoProcessor


def encode_answer(q_type: str, q_gt_answer: str):
    if q_type == "descriptive:count":
        return q_gt_answer  # '0', '1', '2', '3', '4', or '5+'

    elif q_type == "descriptive:exist": # 'yes' or 'no'
        if q_gt_answer == "yes":
            return "A"
        elif q_gt_answer == "no":
            return "B"
        else:
            raise ValueError(f"Received unexpected q_gt_answer={q_gt_answer}!")

    elif q_type == "descriptive:query_material": # 'metal' or 'rubber'
        if q_gt_answer == "metal":
            return "A"
        elif q_gt_answer == "rubber":
            return "B"
        else:
            raise ValueError(f"Received unexpected q_gt_answer={q_gt_answer}!")

    elif q_type == "descriptive:query_shape": # 'cube', 'sphere', or 'cylinder'
        if q_gt_answer == "cube":
            return "A"
        elif q_gt_answer == "sphere":
            return "B"
        elif q_gt_answer == "cylinder":
            return "C"
        else:
            raise ValueError(f"Received unexpected q_gt_answer={q_gt_answer}!")

    elif q_type == "descriptive:query_color": # 'gray', 'red', 'blue', 'green', 'brown', 'cyan', 'purple', or 'yellow'
        if q_gt_answer == "gray":
            return "A"
        elif q_gt_answer == "red":
            return "B"
        elif q_gt_answer == "blue":
            return "C"
        elif q_gt_answer == "green":
            return "D"
        elif q_gt_answer == "brown":
            return "E"
        elif q_gt_answer == "cyan":
            return "F"
        elif q_gt_answer == "purple":
            return "G"
        elif q_gt_answer == "yellow":
            return "H"
        else:
            raise ValueError(f"Received unexpected q_gt_answer={q_gt_answer}!")

    elif "explanatory" in q_type: # 'correct' or 'wrong'
        if q_gt_answer == "correct":
            return "A"
        elif q_gt_answer == "wrong":
            return "B"
        else:
            raise ValueError(f"Received unexpected q_gt_answer={q_gt_answer}!")

    elif q_type == "predictive": # ['correct', 'wrong'] or ['wrong', 'correct']
        if q_gt_answer[0] == 'correct' and q_gt_answer[1] == 'wrong':
            return "A"
        elif q_gt_answer[0] == 'wrong' and q_gt_answer[1] == 'correct':
            return "B"
        else:
            raise ValueError(f"Received unexpected q_gt_answer={q_gt_answer}!")

    elif "counterfactual" in q_type: # 'correct' or 'wrong'
        if q_gt_answer == "correct":
            return "A"
        elif q_gt_answer == "wrong":
            return "B"
        else:
            raise ValueError(f"Received unexpected q_gt_answer={q_gt_answer}!")

    else:
        raise ValueError(f"Received unexpected q_type={q_type}!\n")


def convert_to_prompt(question_data, with_answer=False):
    q_type = question_data["q_type"]
    q_text = question_data["q_text"]
    q_choices = question_data["q_choices"]
    q_gt_answer = question_data["q_gt_answer"]

    prompt  = f"USER: <video>\nAnswer the following question based on the video.\n"
    prompt += f"Question: {q_text}\n"
    # based on the question type, the prompt might need to be slightly adjusted...
    if q_type == "descriptive:count":
        prompt += f"ASSISTANT: Answer: "
        if with_answer:
            prompt += encode_answer(q_type, q_gt_answer)

    elif q_type == "descriptive:exist":
        prompt += f"The choices are: A. yes; B. no;\nASSISTANT: Answer: "
        if with_answer:
            prompt += encode_answer(q_type, q_gt_answer)

    elif q_type == "descriptive:query_material":
        prompt += f"The choices are: A. metal; B. rubber;\nASSISTANT: Answer: "
        if with_answer:
            prompt += encode_answer(q_type, q_gt_answer)

    elif q_type == "descriptive:query_shape":
        prompt += f"The choices are: A. cube; B. sphere; C. cylinder;\nASSISTANT: Answer: "
        if with_answer:
            prompt += encode_answer(q_type, q_gt_answer)

    elif q_type == "descriptive:query_color":
        prompt += f"The choices are: A. gray; B. red; C. blue; D. green; E. brown; F. cyan; G. purple; H. yellow;\nASSISTANT: Answer: "
        if with_answer:
            prompt += encode_answer(q_type, q_gt_answer)

    elif "explanatory" in q_type:
        prompt += f"The event is: {q_choices[0]}\n"
        prompt += f"The choices are: A. correct; B. wrong;\nASSISTANT: Answer: "
        if with_answer:
            prompt += encode_answer(q_type, q_gt_answer)

    elif q_type == "predictive":
        prompt += f"The events are: A. {q_choices[0]}; B. {q_choices[1]};\nASSISTANT: Answer: "
        if with_answer:
            prompt += encode_answer(q_type, q_gt_answer)

    elif "counterfactual" in q_type:
        prompt += f"The event is: {q_choices[0]}\n"
        prompt += f"The choices are: A. correct; B. wrong;\nASSISTANT: Answer: "
        if with_answer:
            prompt += encode_answer(q_type, q_gt_answer)

    else:
        raise ValueError(f"Received unexpected q_type={q_type}!\n")
    return prompt



class TrainCollate():
    def __init__(self, processor, lightning_mode=False, lm_with_attention=False):
        self.processor = processor
        self.lightning_mode = lightning_mode
        self.lm_with_attention = lm_with_attention

    def __call__(self, samples):
        vq_pairs = []
        texts = []
        videos = []
        gt_answers = []
        human_gaze = []

        for qdata, vdata, adata in samples:
            vq_pairs.append((qdata["v_no"], qdata["q_no"]))
            prompt = convert_to_prompt(question_data=qdata, with_answer=True)
            texts.append(prompt)
            clip = vdata["frames"]
            videos.append(clip)
            gt_answers.append(encode_answer(qdata["q_type"], qdata["q_gt_answer"]))
            human_gaze.append(adata["gaze"])

        batch = self.processor(text=texts, videos=videos, padding=True, return_tensors='pt')

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values_videos = batch["pixel_values_videos"]
        labels = batch["labels"]
        if self.lightning_mode:
            if self.lm_with_attention:
                return input_ids, attention_mask, pixel_values_videos, labels, human_gaze_to_tensor(human_gaze)
            return input_ids, attention_mask, pixel_values_videos, labels
        return input_ids, attention_mask, pixel_values_videos, labels, (vq_pairs, texts, gt_answers, human_gaze)


class EvalCollate():
    def __init__(self, processor, lightning_mode=False, lm_with_attention=False):
        self.processor = processor
        self.lightning_mode = lightning_mode
        self.lm_with_attention = lm_with_attention

    def __call__(self, samples):
        vq_pairs = []
        texts = []
        videos = []
        gt_answers = []
        human_gaze = []

        for qdata, vdata, adata in samples:
            vq_pairs.append((qdata["v_no"], qdata["q_no"]))
            prompt = convert_to_prompt(question_data=qdata, with_answer=False)
            texts.append(prompt)
            clip = vdata["frames"]
            videos.append(clip)
            gt_answers.append(encode_answer(qdata["q_type"], qdata["q_gt_answer"]))
            human_gaze.append(adata["gaze"])

        batch = self.processor(text=texts, videos=videos, padding=True, return_tensors='pt')

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values_videos = batch["pixel_values_videos"]
        if self.lightning_mode:
            if self.lm_with_attention:
                return input_ids, attention_mask, pixel_values_videos, gt_answers, human_gaze_to_tensor(human_gaze)
            return input_ids, attention_mask, pixel_values_videos, gt_answers
        return input_ids, attention_mask, pixel_values_videos, (vq_pairs, texts, gt_answers, human_gaze)



def human_gaze_to_tensor(
    human_gaze,
    *,
    dtype=torch.float32,
    device=None,
    pad_value=float('nan'),
    return_mask=False,
):
    """
    Convert nested list human_gaze -> torch tensor [1, num_frames, max_recordings, 2].

    Expected input layout:
        human_gaze = [
            [   # sample 0 (batch size is always 1)
                [(x,y), (x,y), ...],  # frame 0 recordings
                [(x,y), ...],         # frame 1 recordings
                ...
            ]
        ]

    - Frames can have different numbers of recordings; shorter ones are padded with `pad_value`.
    - If `return_mask=True`, also returns a bool mask of shape [1, num_frames, max_recordings]
      where True marks valid (non-padded) recordings.
    """
    assert isinstance(human_gaze, list) and len(human_gaze) == 1, \
        "Expected outer list (batch) of length 1."

    sample = human_gaze[0]                   # list over frames
    num_frames = len(sample)
    rec_counts = [len(rec_list) for rec_list in sample]
    max_rec = max(rec_counts) if num_frames > 0 else 0

    out = torch.full((1, num_frames, max_rec, 2), pad_value, dtype=dtype, device=device)
    mask = torch.zeros((1, num_frames, max_rec), dtype=torch.bool, device=device) if return_mask else None

    for f_idx, rec_list in enumerate(sample):
        if not rec_list:
            continue
        t = torch.as_tensor(rec_list, dtype=dtype, device=device)  # [n_rec, 2]
        n = t.shape[0]
        out[0, f_idx, :n, :] = t
        if return_mask:
            mask[0, f_idx, :n] = True

    return (out, mask) if return_mask else out


def get_humaneyecue_dataloader(humaneyecue_config_file, split, video_sampling_method, video_num_frames, attention_type, batch_size, processor, lightning_mode=False, lm_with_attention=False):
    assert split in ['train', 'valid', 'test', 'trainval', 'all'], f"Received unexpected split={split}! Options are: ('train', 'valid', 'test', 'trainval', 'all').\n"

    data = HumanEyeCue(humaneyecue_config_file, split, video_sampling_method, video_num_frames, attention_type)
    if split in ['train', 'trainval']:
        loader = torch.utils.data.DataLoader(data, collate_fn=TrainCollate(processor, lightning_mode, lm_with_attention), batch_size=batch_size, shuffle=True, num_workers=4)
    elif split in ['valid', 'test', 'all']:
        loader = torch.utils.data.DataLoader(data, collate_fn=EvalCollate(processor, lightning_mode, lm_with_attention), batch_size=batch_size, shuffle=False, num_workers=4)
    return loader




if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained('./models/video_llava/video_llava_hf_model/Video-LLaVA-7B-hf')
    processor.tokenizer.padding_side = "right"

    train_loader = get_humaneyecue_dataloader(humaneyecue_config_file='data/model_inputs/model_config.json',
                                           split='test', video_sampling_method='uniform', video_num_frames=8, attention_type='spatial',
                                           batch_size=1, processor=processor, lightning_mode=True, lm_with_attention=True)


    # for batch_idx, sample in enumerate(train_loader):
    #     print(f"Human gaze tensor shape: {sample[-1].shape}")
    #     print(f"Number of NaNs in human gaze tensor: {np.isnan(sample[-1].numpy()).sum()}")
    #     print(f"Batch idx: {batch_idx}")

    dataiter = iter(train_loader)
    sample = next(dataiter)

    # from humaneyecue_helper_functions import calculate_pixels_for_angle, create_human_saliency_maps_from_gaze, downsample_human_saliency_maps
    # ## estimate the standard deviation of the Gaussian fovea in pixels for videos
    # stdev_gaussian_fovea_screen = calculate_pixels_for_angle(viewing_distance_cm = 55, visual_angle_deg = 2, screen_resolution = tuple([1080, 1920]),  screen_diagonal_inches = 24)
    # stdev_gaussian_fovea_videos = (stdev_gaussian_fovea_screen * 480) / 1920

    # ## create human saliency maps for the videos in the batch
    # human_gaze = sample[-1]  # shape: (batch_size, num_frames, num_human_gaze_points, 2)
    # human_saliency_maps = create_human_saliency_maps_from_gaze(human_gaze=human_gaze.detach().cpu().numpy(), saliency_map_size=tuple([320, 480]), sigma_px=stdev_gaussian_fovea_videos) # shape: (batch_size, num_frames, saliency_map_height, saliency_map_width)
    # human_saliency_maps = torch.tensor(human_saliency_maps).to(human_gaze.device)  # convert to tensor and move to the same device as the input batch

    # ## downsample saliency maps to match the feature map size
    # downsampled_human_saliency_maps = downsample_human_saliency_maps(human_saliency_maps=human_saliency_maps, target_size=(16, 16), frame_size=(224, 224), use_interpolation=True, nondivisible_strategy="resize")  # shape: (batch_size, num_frames, 16, 16)

    print("Done.")
