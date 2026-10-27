import os
import json
import numpy as np
import torch
import pickle
import argparse
from transformers import BitsAndBytesConfig
from transformers import VideoLlavaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from humaneyecue_dataloader import get_humaneyecue_dataloader
from humaneyecue_helper_functions import calculate_pixels_for_angle, create_human_saliency_maps_from_gaze, downsample_human_saliency_maps, adjust_inputs_based_on_reduced_video_tokens

## Reproducibility
np.random.seed(19)
torch.manual_seed(19)
torch.cuda.manual_seed_all(19)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Configuration and Hyperparameters
model_dir = ""
model_id = "LanguageBind/Video-LLaVA-7B-hf"
USE_QLORA, USE_LORA = True, True
config = {
    'humaneyecue_config_file': 'data/model_inputs/model_config.json',
    'video_sampling_method': 'uniform',
    'video_num_frames': 8,
    'attention_type': 'spatial',
    'attention_threshold': 0.5,

    'max_epochs': 20,
    'learning_rate': 3e-6,
    'num_nodes': 1,
    'warmpup_steps': 50,
    'check_val_every_n_epoch': 1,
    'gradient_clip_val': 1.0,
    'batch_size': 1,
    'accumulate_grad_batches': 16,
    'verbose': True,
    'results_dir': './output/video_llava',

    "data_collection_settings": {
        "video_resolution":          [320, 480],
        "screen_resolution":         [1080, 1920],
        "screen_diagonal_inches":    24,
        "viewing_distance_cm":       55,
        "visual_angle_degrees":      2
    },
}

# Release CLI overrides are applied before loading weights or data.
parser = argparse.ArgumentParser(description="Fine-tune the paper's VQA baseline or gaze variant.")
parser.add_argument('--model', default=model_id)
parser.add_argument('--data-config', default=config['humaneyecue_config_file'])
parser.add_argument('--output-dir', default=config['results_dir'])
parser.add_argument('--epochs', type=int, default=config['max_epochs'])
release_args = parser.parse_args()
model_dir, model_id = '', release_args.model
config.update(humaneyecue_config_file=release_args.data_config, results_dir=release_args.output_dir,
              max_epochs=release_args.epochs)
L.seed_everything(19, workers=True)

## Load processor
processor = AutoProcessor.from_pretrained(model_dir + model_id)
processor.tokenizer.padding_side = "right"

## Load model
# Three options for training, from the lowest precision training to the highest precision training:
# QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.
# Standard LoRA:  model is loaded with standard LoRA adaptations.
# Full Fine-Tuning: no memory optimization are done. In that case Flash Attention is used to speed up training, if hardware supports it.
if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_dir + model_id,
        torch_dtype = torch.float16,
        quantization_config = bnb_config if USE_QLORA else None,
        device_map = "auto"
    )
else:
    # for full fine-tuning, we can speed up the model using Flash Attention
    # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_dir + model_id,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
        device_map="auto",
    )

## Apply PEFT LoRA
def find_all_linear_names(model):
    lora_module_names = []
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords) or 'lm_head' in name: # 'lm_head' is needed for 16-bit
            continue
        if isinstance(module, torch.nn.Linear):
            lora_module_names.append(name)
    return lora_module_names

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.language_model.set_attn_implementation("eager")

## Define PyTorch Lightning Module
class VideoLlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.save_hyperparameters(ignore=['processor', 'model'])
        self.config = self.hparams.config
        self.processor = processor
        self.model = model
        self.batch_size = self.config['batch_size']
        self.attention_th = self.config['attention_threshold']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def train_dataloader(self):
        return get_humaneyecue_dataloader(
            humaneyecue_config_file=self.config['humaneyecue_config_file'],
            split='train',
            video_sampling_method=self.config['video_sampling_method'],
            video_num_frames=self.config['video_num_frames'],
            attention_type=self.config['attention_type'],
            batch_size=self.config['batch_size'],
            processor=self.processor,
            lightning_mode=True, lm_with_attention=True
        )

    def val_dataloader(self):
        return get_humaneyecue_dataloader(
            humaneyecue_config_file=self.config['humaneyecue_config_file'],
            split='valid',
            video_sampling_method=self.config['video_sampling_method'],
            video_num_frames=self.config['video_num_frames'],
            attention_type=self.config['attention_type'],
            batch_size=self.config['batch_size'],
            processor=self.processor,
            lightning_mode=True, lm_with_attention=True
        )

    def get_human_saliency_maps(self, human_gaze):
        ## 1. estimate the standard deviation of the Gaussian fovea in pixels for videos
        stdev_gaussian_fovea_screen = calculate_pixels_for_angle(
            viewing_distance_cm = self.config['data_collection_settings']['viewing_distance_cm'],
            visual_angle_deg = self.config['data_collection_settings']['visual_angle_degrees'],
            screen_resolution = tuple(self.config['data_collection_settings']['screen_resolution']),
            screen_diagonal_inches = self.config['data_collection_settings']['screen_diagonal_inches'])
        stdev_gaussian_fovea_videos = (stdev_gaussian_fovea_screen * self.config['data_collection_settings']['video_resolution'][1]) / self.config['data_collection_settings']['screen_resolution'][1]

        ## 2. create human saliency maps for the videos in the batch
        human_saliency_maps = create_human_saliency_maps_from_gaze(
            human_gaze=human_gaze.detach().cpu().numpy(),
            saliency_map_size=tuple(self.config['data_collection_settings']['video_resolution']),
            sigma_px=stdev_gaussian_fovea_videos) # shape: (batch_size, num_frames, saliency_map_height, saliency_map_width)
        human_saliency_maps = torch.tensor(human_saliency_maps).to(human_gaze.device)  # convert to tensor and move to the same device as the input batch

        ## 3. downsample human saliency maps to match the size of the video frames returned by preprocessor
        B, T, H, W = human_saliency_maps.shape
        target_height, target_width = self.processor.video_processor.crop_size['height'], self.processor.video_processor.crop_size['width']
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

        # ## 3. downsample human saliency maps to match the size of the model's attention maps
        # downsampled_human_saliency_maps = downsample_human_saliency_maps(
        #     human_saliency_maps=human_saliency_maps,
        #     target_size=(16, 16),
        #     frame_size=(224, 224),
        #     use_interpolation=True,
        #     nondivisible_strategy="resize")  # shape: (batch_size, num_frames, 16, 16)
        return downsampled_human_saliency_maps


    def apply_human_saliency_to_input(self, pixel_values_videos, human_gaze):
        human_saliency_maps = self.get_human_saliency_maps(human_gaze)  # shape: (batch_size, num_frames, H, W)
        # expand saliency maps to match the number of channels in pixel_values_videos
        human_saliency_maps_expanded = human_saliency_maps.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # shape: (batch_size, num_frames, 3, H, W)
        # apply saliency maps to pixel values
        pixel_values_videos = pixel_values_videos * human_saliency_maps_expanded
        return pixel_values_videos


    def get_human_saliency_visual_tokens_mask(self, human_gaze, threshold):
        human_saliency_maps = self.get_human_saliency_maps(human_gaze)
        # Average saliency maps over patches
        ksize   = self.model.video_tower.vision_model.embeddings.patch_embedding.kernel_size
        stride  = self.model.video_tower.vision_model.embeddings.patch_embedding.stride
        averaged_human_saliency_maps = torch.nn.functional.avg_pool2d(human_saliency_maps, kernel_size=ksize, stride=stride, ceil_mode=False)
        # Threshold patches based on saliency values
        binary_saliency_maps = (averaged_human_saliency_maps >= threshold).float()
        return binary_saliency_maps



    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values_videos, labels, human_gaze = batch

        # # apply human saliency maps to pixel values
        # pixel_values_videos = self.apply_human_saliency_to_input(pixel_values_videos, human_gaze)

        # prepare human attention visual tokens mask
        self.model.video_tower.vision_model.clear_human_attention()
        human_attention_visual_tokens_mask = self.get_human_saliency_visual_tokens_mask(human_gaze, self.attention_th)
        human_attention_visual_tokens_mask_reshaped = human_attention_visual_tokens_mask.reshape(self.config['batch_size']*self.config['video_num_frames'], -1)
        human_attention_visual_tokens_avg_per_frame = human_attention_visual_tokens_mask_reshaped.count_nonzero().float().item() / human_attention_visual_tokens_mask_reshaped.shape[0]
        self.log('train_avg_video_tokens_kept_per_frame', human_attention_visual_tokens_avg_per_frame, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.model.video_tower.vision_model.store_human_attention(human_attention_visual_tokens_mask_reshaped)
        input_ids, attention_mask = adjust_inputs_based_on_reduced_video_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            human_attention_visual_tokens_mask=human_attention_visual_tokens_mask,
            video_token_id=self.model.config.video_token_id,
            pad_token_id=self.model.config.pad_token_id,
        )
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels,
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values_videos, answers, human_gaze = batch

        # # apply human saliency maps to pixel values
        # pixel_values_videos = self.apply_human_saliency_to_input(pixel_values_videos, human_gaze)

        # prepare human attention visual tokens mask
        self.model.video_tower.vision_model.clear_human_attention()
        human_attention_visual_tokens_mask = self.get_human_saliency_visual_tokens_mask(human_gaze, self.attention_th)
        human_attention_visual_tokens_mask_reshaped = human_attention_visual_tokens_mask.reshape(self.config['batch_size']*self.config['video_num_frames'], -1)
        human_attention_visual_tokens_avg_per_frame = human_attention_visual_tokens_mask_reshaped.count_nonzero().float().item() / human_attention_visual_tokens_mask_reshaped.shape[0]
        self.log('val_avg_video_tokens_kept_per_frame', human_attention_visual_tokens_avg_per_frame, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.model.video_tower.vision_model.store_human_attention(human_attention_visual_tokens_mask_reshaped)
        input_ids, attention_mask = adjust_inputs_based_on_reduced_video_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            human_attention_visual_tokens_mask=human_attention_visual_tokens_mask,
            video_token_id=self.model.config.video_token_id,
            pad_token_id=self.model.config.pad_token_id,
        )
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100


        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            max_new_tokens=50,
            do_sample=False,
        )
        preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        # Compute accuracy
        correct = 0
        total = 0
        for pred, ans in zip(preds, answers):
            model_answer = pred.split("ASSISTANT: Answer: ")[1]
            if len(model_answer) > 0 and model_answer.lstrip()[0] == ans:
                correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Compute validation loss
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels,
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return correct


model_module = VideoLlavaModelPLModule(config, processor, model)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')
class SaveModelCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        save_path = os.path.join(config['results_dir'], f"lightning_logs/version_{trainer.logger.version}", f"epoch{trainer.current_epoch+1}")
        pl_module.model.save_pretrained(save_path, from_pt=True)
        print(f"Model checkpoint saved at {save_path}")
    def on_train_end(self, trainer, pl_module):
        final_save_path = os.path.join(config['results_dir'], f"lightning_logs/version_{trainer.logger.version}", "final")
        pl_module.model.save_pretrained(final_save_path, from_pt=True)
        print(f"Final model saved at {final_save_path}")


## Define trainer
trainer = L.Trainer(
        default_root_dir=config['results_dir'],
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        num_sanity_val_steps=1,
        callbacks=[early_stopping_callback, SaveModelCallback()],
        log_every_n_steps=50,
)

trainer.fit(model_module)
print("Training completed.")
