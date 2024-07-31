""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn



@dataclass
class MocaVisionEncoderConfig:
    hf_model_name: Optional[str] = None
    image_input_type: str = None
    patch_size: int = None
    image_size: Union[Tuple[int, int], int] = None
    in_channels: int = None
    q_former: int = None
    # No pre convolution norm: normalization = None
    # LayerNorm: normalization = 1
    # InstanceNorm: normalization = in_channels
    # GroupNorm: normalization = 1 < x < in_channels
    normalization: int = None
    lora: bool = False
    pretrained: Optional[str] = None


@dataclass
class MocaTextDecoderConfig:
    hf_model_name: Optional[str] = None
    hf_tokenizer_name: Optional[str] = None
    hf_model_pretrained: bool = True
    projection_type: str = None
    pretrained: bool = None
    ignore_index: int = -100
    lora: bool = False

@dataclass
class MAEEncoderConfig(MocaVisionEncoderConfig):
    mask_ratio: float = None
    input_noise: float = 0.0

@dataclass
class MAEDecoderConfig:
    hf_model_name: Optional[str] = None
    size_factor: int = None
    normalize_labels: bool = False


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)
        # TODO: Fix this?
        # if isinstance(l, (VisionEncoder)):
        #     # convert text nn.Parameter projections
        #     attr = getattr(l, "text_projection", None)
        #     if attr is not None:
        #         attr.data = attr.data.to(dtype)
        #
        # if isinstance(l, VisionTransformer):
        #     # convert vision nn.Parameter projections
        #     attr = getattr(l, "proj", None)
        #     if attr is not None:
        #         attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)
