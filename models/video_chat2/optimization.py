"""Explicit component freezing and the source cosine scheduler."""
import math


def configure_trainable(model, recipe):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ('lora_' in name or (recipe.qformer_lr is not None and
                                   (name.startswith('qformer.') or name in ('query_tokens', 'extra_query_tokens'))))
    # Frozen upstream components must stay in evaluation mode during fine-tuning.
    model.vision_encoder.eval()
    model.vision_layernorm.eval()
    if recipe.qformer_lr is None:
        model.qformer.eval()


def optimizer_groups(model, recipe):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_qformer = name.startswith('qformer.') or name in ('query_tokens', 'extra_query_tokens')
        lr = recipe.qformer_lr if is_qformer else recipe.lora_lr
        if lr is None:
            raise ValueError(f'Unexpected trainable parameter: {name}')
        wd = 0. if parameter.ndim == 1 or name.endswith('.bias') else recipe.weight_decay
        groups.setdefault((lr, wd), []).append(parameter)
    if not groups:
        raise ValueError('No trainable parameters')
    return [dict(params=params, lr=lr, weight_decay=wd) for (lr, wd), params in groups.items()]


def lr_multiplier(step, steps, recipe):
    warmup = steps / recipe.epochs * recipe.warmup_epochs
    if step < warmup:
        return max(recipe.min_lr_multiplier, step / max(1, warmup))
    progress = min(1., (step-warmup) / max(1, steps-warmup))
    return max(recipe.min_lr_multiplier, .5 * (1 + math.cos(math.pi * progress)))
