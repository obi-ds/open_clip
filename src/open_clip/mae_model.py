"""Moca model"""
import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional
from transformers import AutoConfig


from .model import MAEEncoderConfig, MAEDecoderConfig
from .biogpt_vision import MaskedVisionBioGPTModel, VisionBioGPTModel
from .transformer_decoder import VisionEncoder, MaskedAutoencoderVisionEncoder

_has_transformers = True


class MAE(nn.Module):
    """
    MAE class
    """
    def __init__(
            self,
            encoder_cfg: MAEEncoderConfig,
            decoder_cfg: MAEDecoderConfig,
            **kwargs
    ):
        print('MAE Model')
        super().__init__()
        encoder_cfg = (
            MAEEncoderConfig(**encoder_cfg)
            if isinstance(encoder_cfg, dict) else encoder_cfg
        )

        decoder_cfg = (
            MAEDecoderConfig(**decoder_cfg)
            if isinstance(decoder_cfg, dict) else decoder_cfg
        )

        print('Encoder Config: ', encoder_cfg)
        print('Decoder Config: ', decoder_cfg)

        if encoder_cfg.mask_ratio is None:
            raise ValueError('Mask Ratio not set')

        encoder = self.get_encoder(
            image_input_type=encoder_cfg.image_input_type,
            image_size=encoder_cfg.image_size,
            patch_size=encoder_cfg.patch_size,
            in_channels=encoder_cfg.in_channels,
            model_name_or_path=encoder_cfg.hf_model_name,
            normalization=encoder_cfg.normalization,
            mask_ratio=encoder_cfg.mask_ratio
        )

        # TODO: We need to replace this correctly
        # I think we use just the vision biogpt model
        decoder = self.get_decoder(
            model_name_or_path=decoder_cfg.hf_model_name,
            size_factor=decoder_cfg.size_factor
        )

        self._masked_auto_encoder_vision_encoder = MaskedAutoencoderVisionEncoder(
            encoder=encoder,
            decoder=decoder,
            image_input_type=encoder_cfg.image_input_type,
            patch_size=encoder_cfg.patch_size,
            in_channels=encoder_cfg.in_channels,
        )

        self._normalize_labels = decoder_cfg.normalize_labels

        # Set these to None - so that it works with the existing open_clip implementation
        self.visual = None
        self.text = None

        # Added to test denoising MAE
        self.input_noise = encoder_cfg.input_noise

    @staticmethod
    def get_encoder(
            image_input_type: str,
            image_size: int,
            patch_size: int,
            model_name_or_path: str,
            in_channels: int,
            normalization: Optional[int],
            mask_ratio: float
    ):
        """
        Return the vision encoder object

        Args:
            image_input_type:
            image_size:
            patch_size:
            model_name_or_path:
            in_channels:
            normalization:
            mask_ratio:

        Returns:

        """
        # TODO: Currently we only support the BioGPT architecture
        #       for other architectures - we need to modify the attention mask accordingly
        #       and create another subclass
        config = AutoConfig.from_pretrained(model_name_or_path)
        transformer = MaskedVisionBioGPTModel(config=config, mask_ratio=mask_ratio)
        return VisionEncoder(
            image_input_type=image_input_type,
            image_size=image_size,
            patch_size=patch_size,
            config=config,
            transformer=transformer,
            in_channels=in_channels,
            normalization=normalization
        )

    @staticmethod
    def get_decoder(
            model_name_or_path: str,
            size_factor: int
    ):
        """
        Return the vision encoder object

        Args:
            model_name_or_path:
            size_factor:

        Returns:

        """
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_attention_heads = config.num_attention_heads // size_factor
        config.num_hidden_layers = config.num_hidden_layers // size_factor
        config.intermediate_size = config.intermediate_size // size_factor
        config.hidden_size = config.hidden_size // size_factor
        return VisionBioGPTModel(config=config)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        """
        Gradient checkpointing

        Args:
            enable:

        Returns:

        """
        self.visual.set_grad_checkpointing(enable)

    def forward(
            self,
            images,
            texts=None,
    ):

        if self.input_noise > 0:
            std = images.std(dim=-1, keepdim=True)
            noise_scale = self.input_noise * std
            images = images + torch.randn_like(images) * noise_scale

        hidden_states, mask, ids_restore = self._masked_auto_encoder_vision_encoder.forward_encoder(x=images)
                
        #encoded = self.dropout(hidden_states)
        predictions = self._masked_auto_encoder_vision_encoder.forward_decoder(hidden_states, ids_restore)
        labels = self._masked_auto_encoder_vision_encoder.patchify(images, normalize_labels=self._normalize_labels)
        reconstructions = self._masked_auto_encoder_vision_encoder.unpatchify_ecg(predictions)

        # # Calculate MSE loss only on masked tokens
        # mse_loss = (predictions - labels) ** 2
        # mse_loss = mse_loss.mean(dim=-1)  # [N, L], mean loss per patch
        # mse_loss = (mse_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        # # TODO this is recursively called
        # #consistency_loss = self.consistency_loss(images)
        # amp_reg = self.amplitude_regularization(images, reconstructions)
        # freq_reg = self.frequency_regularization(images, reconstructions)
        # tv_reg = self.total_variation_regularization(reconstructions)

        return {
            'hidden_states': hidden_states,
            'reconstructions': reconstructions,
            'images': images,
            'predictions': predictions,
            'labels': labels,
            'mask': mask,
        }
    