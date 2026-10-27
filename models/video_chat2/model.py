"""VideoChat2 Vicuna architecture with the original checkpoint parameter names.

Adapted from OpenGVLab/Ask-Anything and the HumanEyeCue experimental source.
See THIRD_PARTY.md. Only the instruction-tuned, eager-attention path is retained.
"""
import contextlib
import logging

import torch
from torch import nn
from transformers import BertConfig, BertTokenizer, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from .backbones.Qformer import BertLMHeadModel
from .backbones.modeling_llama import LlamaForCausalLM
from .backbones.vit import build_vit
from .config import vision_config

logger = logging.getLogger(__name__)


def load_weights(model, filename, required_prefixes=(), allowed_missing=None):
    """Accept raw original state dicts and local training payloads; reject bad matches."""
    payload = torch.load(filename, map_location='cpu', weights_only=False)
    state = payload.get('model', payload)
    if not isinstance(state, dict) or not state:
        raise ValueError(f'Empty or invalid checkpoint: {filename}')
    expected = model.state_dict()
    unexpected = set(state) - set(expected)
    if unexpected:
        raise ValueError(f'Unexpected checkpoint keys in {filename}: {sorted(unexpected)[:8]}')
    for prefix in required_prefixes:
        if not any(key.startswith(prefix) for key in state):
            raise ValueError(f'Checkpoint {filename} has no {prefix} weights')
    if allowed_missing is not None:
        missing = set(allowed_missing) - set(state)
        if missing:
            raise ValueError(f'Checkpoint is missing trained parameters: {sorted(missing)[:8]}')
    result = model.load_state_dict(state, strict=False)
    logger.info('Loaded %s (%d tensors, %d omitted)', filename, len(state), len(result.missing_keys))
    return payload


class VideoChat2(nn.Module):
    def __init__(self, resources, recipe):
        super().__init__()
        self.recipe = recipe
        self.use_mask = recipe.gaze
        self.max_txt_len = 512
        self.tokenizer = BertTokenizer.from_pretrained(resources['bert_model_path'],
                                                       truncation_side='left', local_files_only=True)
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        vision = vision_config(recipe)
        self.vision_encoder = build_vit(vision)
        vision_width = vision.vision_encoder.encoder_embed_dim
        self.vision_layernorm = nn.LayerNorm(vision_width, eps=1e-12)
        config = BertConfig.from_pretrained(resources['bert_model_path'], local_files_only=True)
        config.encoder_width = vision_width
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        config.query_length = 32
        config.hidden_dropout_prob = .1
        config.attention_probs_dropout_prob = .1
        config.drop_path_list = torch.linspace(0, .2, config.num_hidden_layers).tolist()
        self.qformer = BertLMHeadModel(config)
        self.query_tokens = nn.Parameter(torch.zeros(1, 32, config.hidden_size))
        nn.init.normal_(self.query_tokens, std=config.initializer_range)
        self.qformer.resize_token_embeddings(len(self.tokenizer))
        self.qformer.cls = None
        # The original UMT payload may include a pretraining head not used here.
        payload = torch.load(resources['vit_blip_model_path'], map_location='cpu', weights_only=False)
        state = payload.get('model', payload)
        expected = self.state_dict()
        selected = {key: value for key, value in state.items() if key in expected}
        if not all(any(k.startswith(prefix) for k in selected) for prefix in ('vision_encoder.', 'qformer.')):
            raise ValueError('UMT checkpoint must contain vision encoder and Q-Former weights')
        self.load_state_dict(selected, strict=False)
        self.extra_num_query_token = 64
        self.extra_query_tokens = nn.Parameter(torch.zeros(1, 64, config.hidden_size))
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(resources['llama_model_path'],
                                                            use_fast=False, local_files_only=True)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        # float32 on CPU also permits tiny-model verification without CUDA.
        self.llama_model = LlamaForCausalLM.from_pretrained(
            resources['llama_model_path'], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True)
        self.llama_model = get_peft_model(self.llama_model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=recipe.lora_r,
            lora_alpha=recipe.lora_alpha, lora_dropout=recipe.lora_dropout))
        self.llama_proj = nn.Linear(config.hidden_size, self.llama_model.config.hidden_size)
        load_weights(self, resources['videochat2_model_path'],
                     required_prefixes=('qformer.', 'llama_proj.', 'llama_model.'))
        self.llama_model.config.use_cache = False
        self.last_keep_indices = None
        self.last_token_counts = None

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.float16) if next(self.parameters()).is_cuda else contextlib.nullcontext()

    def train(self, mode=True):
        super().train(mode)
        self.vision_encoder.eval()
        self.vision_layernorm.eval()
        if self.recipe.qformer_lr is None:
            self.qformer.eval()
        return self

    def encode_img(self, video, instruction='', gaze=None):
        if video.shape[:2] != (1, self.recipe.frames):
            raise ValueError('Paper presets require batch size one and eight frames')
        if not self.use_mask:
            gaze = None
        with self.autocast():
            features = self.vision_encoder(video.permute(0, 2, 1, 3, 4), False, gaze)
            encoder = self.vision_encoder.encoder
            self.last_keep_indices = encoder.last_keep_indices
            self.last_token_counts = (encoder.last_token_counts[0] if self.use_mask else
                                      torch.full((self.recipe.frames,), 196, device=video.device))
            features = features[0] if self.use_mask else features.flatten(1, 2)
            features = self.vision_layernorm(features)
            queries = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            text = self.tokenizer([instruction], padding='longest', truncation=True,
                                  max_length=self.max_txt_len, return_tensors='pt').to(video.device)
            output = self.qformer.bert(
                text.input_ids, attention_mask=torch.cat([
                    torch.ones(queries.shape[:2], dtype=torch.long, device=video.device), text.attention_mask], dim=1),
                query_embeds=queries, encoder_hidden_states=features,
                encoder_attention_mask=torch.ones(features.shape[:2], dtype=torch.long, device=video.device),
                return_dict=True)
            return self.llama_proj(output.last_hidden_state[:, :96])

    def context_embeddings(self, prefix, visual):
        before, after = prefix.split('<VideoHere>') if '<VideoHere>' in prefix else prefix.split('</Video>', 1)
        if '<VideoHere>' not in prefix:
            after = '</Video>' + after
        tokenizer = self.llama_tokenizer
        embed = self.llama_model.get_input_embeddings()
        parts = [tokenizer(text, return_tensors='pt', add_special_tokens=(i == 0)).input_ids.to(visual.device)
                 for i, text in enumerate((before, after))]
        return torch.cat([embed(parts[0]), visual, embed(parts[1])], dim=1)

    def forward(self, video, prefix, answer, gaze=None):
        visual = self.encode_img(video, '', gaze)
        context = self.context_embeddings(prefix, visual)
        tokens = self.llama_tokenizer(answer, return_tensors='pt', add_special_tokens=False).input_ids.to(video.device)
        # Answer-only loss. Unlike the legacy trainer, BOS is an embedding, not a scalar ID.
        max_length = self.max_txt_len + visual.shape[1]
        if context.shape[1] + tokens.shape[1] > max_length:
            raise ValueError('Question and answer exceed the 512-text-token training limit')
        inputs = torch.cat([context, self.llama_model.get_input_embeddings()(tokens)], dim=1)
        labels = torch.cat([torch.full(context.shape[:2], -100, dtype=torch.long, device=video.device), tokens], dim=1)
        with self.autocast():
            return self.llama_model(inputs_embeds=inputs, attention_mask=torch.ones_like(labels),
                                    labels=labels, return_dict=True, use_cache=False).loss

    @torch.no_grad()
    def predict(self, video, prompt, gaze=None):
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopSeparator(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return any(input_ids[0, -len(ids):].tolist() == ids for ids in ([835], [2277, 29937]))

        visual = self.encode_img(video, '', gaze)
        inputs = self.context_embeddings(prompt, visual)
        with self.autocast():
            result = self.llama_model.generate(
                inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=video.device),
                max_new_tokens=100, min_length=1, do_sample=False, num_beams=1, use_cache=True,
                stopping_criteria=StoppingCriteriaList([StopSeparator()]),
                pad_token_id=self.llama_tokenizer.pad_token_id)
        text = self.llama_tokenizer.decode(result[0], skip_special_tokens=True)
        return '(' + text.split('###')[0].strip().split('\n')[0]
