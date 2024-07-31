"""Moca model"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft

from typing import Optional
from transformers import AutoConfig

import matplotlib.pyplot as plt

from .model import MAEEncoderConfig, MAEDecoderConfig
from .biogpt_vision import MaskedVisionBioGPTModel, VisionBioGPTModel
from .transformer_decoder import VisionEncoder, MaskedAutoencoderVisionEncoder

import numpy as np
import scipy.signal

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

        self.sampling_rate = 250  # Make sure this is set correctly

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

    def add_baseline_wander(self, images, frequency_range=(0.01, 0.5), amplitude_range=(0.05, 200)):
        if torch.rand(1) > 0.5:
            return images
        
        batch_size, num_leads, signal_length = images.shape
        t = torch.linspace(0, signal_length / self.sampling_rate, signal_length, device=images.device)
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

    def log_visualizations(self, original_images, noisy_images, cleaned_images, reconstructions, step):
        num_samples = min(2, original_images.shape[0])
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
        fig.suptitle(f'Samples at step {step}')

        for i in range(num_samples):
            for j, (title, image) in enumerate([
                ('Original', original_images[i]),
                ('Noisy', noisy_images[i]),
                ('Cleaned', cleaned_images[i]),
                ('Reconstructed', reconstructions[i])
            ]):
                ax = axes[i, j] if num_samples > 1 else axes[j]
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

        if self.training and self.add_noise:
            images_noisy = self.add_ecg_noise(images)
        else:
            images_noisy = images

        # further denoise the oringinal images
        images_cleaned = denoise_ecg(images, self.sampling_rate)

        hidden_states, mask, ids_restore = self._masked_auto_encoder_vision_encoder.forward_encoder(x=images_noisy)
        predictions = self._masked_auto_encoder_vision_encoder.forward_decoder(hidden_states, ids_restore)
        labels = self._masked_auto_encoder_vision_encoder.patchify(images_cleaned, normalize_labels=self._normalize_labels)
        reconstructions = self._masked_auto_encoder_vision_encoder.unpatchify_ecg(predictions)

        return {
            'hidden_states': hidden_states,
            'reconstructions': reconstructions,
            'images': images_orig,
            'images_noisy': images_noisy,
            'images_cleaned': images_cleaned,
            'predictions': predictions,
            'labels': labels,
            'mask': mask,
        }

def denoise_ecg(signal, sampling_rate=250, low_cutoff_freq=0.5, high_cutoff_freq=125, median_window_size=0.2):
    """
    Remove baseline wander, high-frequency noise, and motion artifacts using filters implemented with PyTorch.
    Can be run on GPU if input signal is on GPU.
    
    Args:
    signal (torch.Tensor): Input ECG signal of shape (batch_size, num_channels, signal_length)
    sampling_rate (float): Sampling rate of the ECG signal in Hz
    low_cutoff_freq (float): Low cutoff frequency for high-pass filter (removes baseline wander)
    high_cutoff_freq (float): High cutoff frequency for low-pass filter (removes high-frequency noise)
    median_window_size (float): Window size in seconds for median filter (removes motion artifacts)
    
    Returns:
    torch.Tensor: Denoised ECG signal
    """
    device = signal.device
    batch_size, num_channels, signal_length = signal.shape
    
    # # Step 1: Remove motion artifacts using median filter
    # window_samples = int(median_window_size * sampling_rate)
    # if window_samples % 2 == 0:
    #     window_samples += 1  # Ensure odd window size for median filter
    
    # # Pad the signal for median filtering
    # pad_size = window_samples // 2
    # padded_signal = torch.nn.functional.pad(signal, (pad_size, pad_size), mode='reflect')
    
    # # Apply median filter
    # motion_filtered = torch.median(padded_signal.unfold(-1, window_samples, 1), dim=-1)[0]
    
    # # Subtract motion artifacts from the original signal
    # signal = signal - motion_filtered
    
    # Step 2: Remove baseline wander and high-frequency noise
    # Create frequency array
    freqs = fft.fftfreq(signal_length, d=1/sampling_rate).to(device)
    
    # Create band-pass filter
    # TODO only using baseline wander filter for now
    band_pass_filter = ((torch.abs(freqs) > low_cutoff_freq) & (torch.abs(freqs) < high_cutoff_freq)).float()
    #band_pass_filter = ((torch.abs(freqs) > low_cutoff_freq)).float()

    # Perform FFT
    signal_fft = fft.fft(signal, dim=-1)
    
    # Apply filter in frequency domain
    filtered_signal_fft = signal_fft * band_pass_filter.unsqueeze(0).unsqueeze(0)
    
    # Perform inverse FFT
    filtered_signal = fft.ifft(filtered_signal_fft, dim=-1).real
    
    return filtered_signal

# def reduce_high_frequency_noise(signal, sampling_rate=250, cutoff_freq=100):
#     """
#     Reduce high-frequency noise using a low-pass filter.
#     """
#     nyquist_freq = 0.5 * sampling_rate
#     normalized_cutoff = cutoff_freq / nyquist_freq
    
#     # Create a low-pass filter
#     sos = scipy.signal.butter(5, normalized_cutoff, btype='low', analog=False, output='sos')
    
#     # Apply the filter
#     filtered_signal = scipy.signal.sosfiltfilt(sos, signal, axis=-1)
    
#     return filtered_signal

# def remove_powerline_interference(signal, sampling_rate=250, notch_freq=50, quality_factor=30):
#     """
#     Remove powerline interference using a notch filter.
#     """
#     nyquist_freq = 0.5 * sampling_rate
#     normalized_freq = notch_freq / nyquist_freq
    
#     # Create a notch filter
#     b, a = scipy.signal.iirnotch(normalized_freq, quality_factor)
    
#     # Apply the filter
#     filtered_signal = scipy.signal.filtfilt(b, a, signal, axis=-1)
    
#     return filtered_signal

# def remove_motion_artifacts(signal, sampling_rate=250, window_size=0.2):
#     """
#     Remove motion artifacts using a median filter.
#     """
#     # Convert window size from seconds to samples
#     window_samples = int(window_size * sampling_rate)
#     if window_samples % 2 == 0:
#         window_samples += 1  # Ensure odd window size for median filter
    
#     # Apply median filter
#     filtered_signal = scipy.signal.medfilt(signal, kernel_size=[1, 1, window_samples])
    
#     return filtered_signal