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
