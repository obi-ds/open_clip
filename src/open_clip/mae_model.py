"""Moca model"""
import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional
from transformers import AutoConfig

import matplotlib.pyplot as plt

from .model import MAEEncoderConfig, MAEDecoderConfig
from .biogpt_vision import MaskedVisionBioGPTModel, VisionBioGPTModel
from .transformer_decoder import VisionEncoder, MaskedAutoencoderVisionEncoder

import numpy as np

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

    def add_baseline_wander(self, images, frequency_range=(0.01, 0.5), amplitude_range=(0.05, 200), sampling_rate=250):
        if torch.rand(1) < 0.5:
            return images
        
        batch_size, num_leads, signal_length = images.shape
        t = torch.linspace(0, signal_length / sampling_rate, signal_length, device=images.device)
        frequencies = torch.rand(batch_size, 1, 1, device=images.device) * (frequency_range[1] - frequency_range[0]) + frequency_range[0]
        amplitudes = torch.rand(batch_size, 1, 1, device=images.device) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
        
        wander = amplitudes * torch.sin(2 * np.pi * frequencies * t.unsqueeze(0))
        return images + wander.expand_as(images)

    def add_powerline_interference(self, images, frequency=50, amplitude=20):
        t = torch.linspace(0, 1, images.shape[-1], device=images.device)
        interference = amplitude * torch.sin(2 * np.pi * frequency * t)
        return images + interference.unsqueeze(0).unsqueeze(0)

    def add_muscle_artifacts(self, images, amplitude=100):
        artifacts = torch.randn_like(images) * amplitude
        mask = torch.rand_like(images) < 0.05  # 10% of timepoints
        return images + artifacts * mask

    def add_noise(self, images, noise_scale):
        std = images.std(dim=-1, keepdim=True)
        noise_scale = noise_scale * std
        mask = torch.rand_like(images) < 0.3  # 10% of timepoints
        return images + torch.randn_like(images) * noise_scale * mask

    def add_ecg_noise(self, images):
        images = self.add_baseline_wander(images)
        images = self.add_powerline_interference(images)
        images = self.add_muscle_artifacts(images)
        images = self.add_noise(images, noise_scale=self.input_noise)
        return images

    def log_visualizations(self, original_images, noisy_images, reconstructions, step):
        num_samples = min(2, original_images.shape[0])
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 10))
        fig.suptitle(f'Samples at step {step}')

        for i in range(num_samples):
            for j, (title, image) in enumerate([
                ('Original', original_images[i]),
                ('Noisy', noisy_images[i]),
                ('Reconstructed', reconstructions[i])
            ]):
                ax = axes[i, j]
                ax.plot(image[0].cpu().detach().numpy(), label='Lead 1')
                ax.plot(image[1].cpu().detach().numpy(), label='Lead 2')
                ax.set_title(f'Sample {i+1} - {title}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude')
                ax.legend()

        plt.tight_layout()

        return fig

    def forward(
            self,
            images,
            texts=None,
    ):

        images_orig = images
        if self.training and self.input_noise > 0:
            images = self.add_ecg_noise(images)

        hidden_states, mask, ids_restore = self._masked_auto_encoder_vision_encoder.forward_encoder(x=images)
                
        predictions = self._masked_auto_encoder_vision_encoder.forward_decoder(hidden_states, ids_restore)
        labels = self._masked_auto_encoder_vision_encoder.patchify(images, normalize_labels=self._normalize_labels)
        reconstructions = self._masked_auto_encoder_vision_encoder.unpatchify_ecg(predictions)

        return {
            'hidden_states': hidden_states,
            'reconstructions': reconstructions,
            'images': images_orig,
            'noisy_images': images,
            'predictions': predictions,
            'labels': labels,
            'mask': mask,
        }
    