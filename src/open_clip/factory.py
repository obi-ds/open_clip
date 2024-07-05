import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from .loss import (
    MoCaLoss, MoCaFocalLoss, MoCaZLoss
)
from .model import convert_weights_to_lp
from .moca_model import MoCa
from .tokenizer import HFTokenizer
from .transformer_decoder import MaskedAutoencoderVisionEncoder


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(
        model_name: str = '',
        context_length: Optional[int] = None,
):
    config = get_model_config(model_name)
    assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get('text_cfg', {})

    tokenizer = HFTokenizer(
        text_config['hf_tokenizer_name'],
        context_length=context_length,
    )

    return tokenizer


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = model_cfg or get_model_config(model_name)
    if model_cfg is not None:
        logging.info(f'Loaded {model_name} model config.')
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    if force_patch_dropout is not None:
        # override the default patch dropout value
        model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

    if force_image_size is not None:
        # override model config's image size
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
    if 'moca' in model_name:
        model = MoCa(
            **model_cfg,
    )
    elif 'mae' in model_name:
        model = MaskedAutoencoderVisionEncoder(
            **model_cfg,
        )
    else:
        raise ValueError(f'Invalid model name: {model_name}')

    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device)
        convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    pretrained_loaded = False
    if pretrained:
        checkpoint_path = pretrained

        if checkpoint_path:
            logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            load_checkpoint(model, checkpoint_path)
        else:
            error_str = (
                f'Pretrained weights ({pretrained}) not found for model {model_name}.'
            )
            logging.warning(error_str)
            raise RuntimeError(error_str)
        pretrained_loaded = True

    if require_pretrained and not pretrained_loaded:
        # callers of create_model_from_pretrained always expect pretrained weights
        raise RuntimeError(
            f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    return model


def create_loss(args):
    if args.loss_function == 'focal':
        return MoCaFocalLoss(ignore_index=-100)
    elif args.loss_function == 'lm':
        return MoCaLoss(ignore_index=-100)
    elif args.loss_function == 'lm_z':
        return MoCaZLoss(ignore_index=-100, penalty_weight=1e-4)
    else:
        raise ValueError(f'Invalid loss function: {args.loss_function}')


def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        **model_kwargs,
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        **model_kwargs,
    )

    preprocess_train = lambda x: x
    preprocess_val = lambda x: x

    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_transform: bool = True,
        cache_dir: Optional[str] = None,
        **model_kwargs,
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        cache_dir=cache_dir,
        require_pretrained=True,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = lambda x: x

    return model, preprocess
