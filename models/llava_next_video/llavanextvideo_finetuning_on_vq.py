import os
import json
import torch
import numpy as np
import pickle
import argparse
from transformers import BitsAndBytesConfig
from transformers import LlavaNextVideoForConditionalGeneration, AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from humaneyecue_dataloader import get_humaneyecue_dataloader
# from humaneyecue_helper_functions import infer_text_video_spans, text_pov_frame_maps, compute_normalized_scanpath_saliency

## Reproducibility
np.random.seed(19)
torch.manual_seed(19)
torch.cuda.manual_seed_all(19)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Configuration and Hyperparameters
model_dir = ""
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
USE_QLORA, USE_LORA = True, True
config = {
    'humaneyecue_config_file': 'data/model_inputs/model_config.json',
    'video_sampling_method': 'uniform',
    'video_num_frames': 16,
    'attention_type': None,

    'max_epochs': 20,
    'learning_rate': 2e-5,
    'lora_r': 8,
    'lora_alpha': 8,
    'lora_dropout': 0.1,
    'num_nodes': 1,
    'warmpup_steps': 50,
    'check_val_every_n_epoch': 1,
    'gradient_clip_val': 1.0,
    'batch_size': 1,
    'accumulate_grad_batches': 16,
    'verbose': True,
    'results_dir': './output/llava_next_video',
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
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_dir + model_id,
        torch_dtype = torch.float16,
        quantization_config = bnb_config if USE_QLORA else None,
        device_map = "auto"
    )
else:
    # for full fine-tuning, we can speed up the model using Flash Attention
    # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
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
    r=config['lora_r'],
    lora_alpha=config['lora_alpha'],
    lora_dropout=config['lora_dropout'],
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.language_model.set_attn_implementation("eager")

## Define PyTorch Lightning Module
class LlavaNextVideoModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.save_hyperparameters(ignore=['processor', 'model'])
        self.config = self.hparams.config
        self.processor = processor
        self.model = model
        self.batch_size = self.config['batch_size']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values_videos, labels = batch
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
        input_ids, attention_mask, pixel_values_videos, answers = batch
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
        return correct

    def train_dataloader(self):
        return get_humaneyecue_dataloader(
            humaneyecue_config_file=self.config['humaneyecue_config_file'],
            split='train',
            video_sampling_method=self.config['video_sampling_method'],
            video_num_frames=self.config['video_num_frames'],
            attention_type=self.config['attention_type'],
            batch_size=self.config['batch_size'],
            processor=self.processor,
            lightning_mode=True
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
            lightning_mode=True
        )

model_module = LlavaNextVideoModelPLModule(config, processor, model)
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
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
