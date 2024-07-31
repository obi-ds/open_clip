from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class MoCaLoss(nn.Module):
    """
    LM Loss function

    """

    def __init__(
            self,
            ignore_index
    ):
        super().__init__()
        self._ignore_index = ignore_index

    def forward(self, logits, labels, weights, output_dict=False):
        """
        Standard LM loss - can assign w eights to tokens if required

        Args:
            logits:
            labels:
            weights:
            output_dict:

        Returns:

        """
        logits = logits[:, :, :].contiguous()
        labels = labels[:, :].contiguous()

        if weights is None:
            caption_loss_obj = nn.CrossEntropyLoss(ignore_index=self._ignore_index, reduction='mean')
            caption_loss = caption_loss_obj(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        else:
            caption_loss_obj = nn.CrossEntropyLoss(ignore_index=self._ignore_index, reduction='none')
            mask = labels != self._ignore_index
            logits = logits[mask]
            labels = labels[mask]
            weights = weights[mask]

            caption_loss = caption_loss_obj(
                logits,
                labels,
            )

            caption_loss = caption_loss * weights.view(-1)
            caption_loss = caption_loss.sum() / weights.sum()

        if output_dict:
            return {"caption_loss": caption_loss}

        return caption_loss


class MoCaZLoss(MoCaLoss):
    """
    LM Loss function with Z-loss regularization

    """

    def __init__(
            self,
            ignore_index,
            penalty_weight,
            reduction: str = 'mean',
    ):
        super().__init__(ignore_index=ignore_index)
        self._ignore_index = ignore_index
        self._penalty_weight = penalty_weight
        self._reduction = reduction

    def forward(self, logits, labels, weights, output_dict=False):
        """
        Standard LM Z loss - can assign weights to tokens if required

        Args:
            logits:
            labels:
            weights:
            output_dict:

        Returns:

        """
        caption_loss = (
            super(MoCaZLoss, self)
            .forward(logits=logits, labels=labels, weights=weights, output_dict=False)
        )

        if self._reduction == 'mean':
            z_loss = torch.pow(torch.logsumexp(logits[labels != self._ignore_index], dim=-1), 2).mean()
        elif self._reduction == 'sum':
            z_loss = torch.pow(torch.logsumexp(logits[labels != self._ignore_index], dim=-1), 2).sum()
        else:
            raise ValueError('Invalid reduction')

        total_loss = caption_loss + (self._penalty_weight * z_loss)

        if output_dict:
            return {"caption_loss": total_loss}

        return total_loss


class MoCaFocalLoss(nn.Module):
    def __init__(
            self,
            ignore_index,
            alpha: Optional[Tensor] = None,
            gamma: float = 1,
            reduction: str = 'mean',
    ):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_index = ignore_index
        self._reduction = reduction
        self.nll_loss = nn.NLLLoss(
            weight=alpha,
            reduction='none',
            ignore_index=ignore_index
        )

    def reduce_loss(self, loss, weights):
        if self._reduction == 'mean':
            if weights is None:
                loss = loss.mean()
            else:
                loss = loss.sum() / weights.sum()
        elif self._reduction == 'sum':
            loss = loss.sum()
        return loss

    def forward(self, logits, labels, weights, output_dict=False):

        logits = logits[:, :, :].contiguous()
        labels = labels[:, :].contiguous()

        mask = labels != self._ignore_index

        logits = logits[mask]
        labels = labels[mask]

        if len(labels) == 0:
            return torch.tensor(0.).to(device=logits.device)

        log_probabilities = F.log_softmax(logits, dim=-1)
        negative_log_pt = self.nll_loss(log_probabilities, labels)

        # Get true class column from each row
        indexes = torch.arange(len(log_probabilities))
        pt = log_probabilities[indexes, labels].exp()

        caption_loss = ((1 - pt) ** self._gamma) * negative_log_pt

        if weights is not None:
            weights = weights[mask]
            caption_loss = caption_loss * weights.view(-1)

        caption_loss = self.reduce_loss(loss=caption_loss, weights=weights)

        if output_dict:
            return {"caption_loss": caption_loss}

        return caption_loss


class MAELoss(nn.Module):
    """
    MAE Loss function

    """

    def __init__(
            self,
            norm_pixel_loss,
    ):
        super().__init__()
        self._norm_pixel_loss = norm_pixel_loss

    def forward(self, hidden_states, predictions, labels, mask, output_dict=False):
        """
        MAE Loss
        """

        if self._norm_pixel_loss:
            mean = labels.mean(dim=-1, keepdim=True)
            var = labels.var(dim=-1, keepdim=True)
            labels = (labels - mean) / (var + 1.e-6) ** .5

        loss = (predictions - labels) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        if output_dict:
            return {"loss": loss}

        return loss


class RegularizedMAELoss(nn.Module):
    """
    MAE Loss function

    """

    def __init__(
            self,
            norm_pixel_loss,
    ):
        super().__init__()
        self._norm_pixel_loss = norm_pixel_loss

    def forward(self,
                hidden_states, reconstructions, images, noisy_images, predictions, labels, mask,
                output_dict=False):
        """
        MAE Loss
        """

        if self._norm_pixel_loss:
            mean = labels.mean(dim=-1, keepdim=True)
            var = labels.var(dim=-1, keepdim=True)
            labels = (labels - mean) / (var + 1.e-6) ** .5

        mse_loss = (predictions - labels) ** 2
        mse_loss = mse_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mse_loss = (mse_loss * mask).sum() / mask.sum()  # mean loss on removed patches

        #amp_reg = self.amplitude_regularization(original=images, reconstructed=reconstructions)
        #freq_reg = self.frequency_regularization(original=images, reconstructed=reconstructions)
        
        # could do this on the hidden states or the reconstructions
        #tv_reg = self.total_variation_regularization(x=hidden_states)
        tv_reg = self.total_variation_regularization(x=reconstructions)
        #amp_reg = 1e-7 * amp_reg
        #freq_reg = 1e-7 * freq_reg
        tv_reg = 1e-4 * tv_reg

        total_loss = mse_loss + tv_reg # amp_reg + freq_reg + tv_reg

        if output_dict:
            return {#"loss": total_loss,
                    'mse_loss': mse_loss,
                    #'amp_reg': amp_reg,
                    #'freq_reg': freq_reg,
                    'tv_reg': tv_reg}

        return total_loss
    
    # def consistency_loss(self, x):
    #     perturbed_x = x + torch.randn_like(x) * 0.1
    #     output1 = self.forward(x)['reconstructions']
    #     output2 = self.forward(perturbed_x)['reconstructions']
    #     return F.mse_loss(output1, output2)

    def amplitude_regularization(self, original, reconstructed):
        original_amp = torch.max(original) - torch.min(original)
        reconstructed_amp = torch.max(reconstructed) - torch.min(reconstructed)
        return F.mse_loss(original_amp, reconstructed_amp)

    def frequency_regularization(self, original, reconstructed):
        def pad_to_power_of_2(x):
            size = x.shape[-1]
            target_size = 2 ** (size - 1).bit_length()
            pad_size = target_size - size
            return F.pad(x, (0, pad_size))

        original_padded = pad_to_power_of_2(original)
        reconstructed_padded = pad_to_power_of_2(reconstructed)

        original_fft = torch.fft.fft(original_padded)
        reconstructed_fft = torch.fft.fft(reconstructed_padded)
        
        return F.mse_loss(torch.abs(original_fft), torch.abs(reconstructed_fft))

    def total_variation_regularization(self, x):
        diff = x[:, :, 1:] - x[:, :, :-1]
        return torch.sum(torch.abs(diff))
    