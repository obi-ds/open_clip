"""Model architectures for decoder only architecture"""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers import (
    AutoConfig,
    BioGptConfig, AutoModel, BioGptModel
)
from peft import LoraModel
from .biogpt_vision import VisionBioGPTModel, MaskedVisionBioGPTModel
from .q_former import BertLMHeadModel


class QFormer(nn.Module):
    """
    Create QFormer model
    """

    def __init__(self, hidden_size: int, num_query_tokens: int):
        super().__init__()
        self._config = self.get_config(hidden_size=hidden_size, num_query_tokens=num_query_tokens)
        self.model = self.get_model(config=self._config)
        self.query_tokens = self.get_query_tokens(num_query_tokens=num_query_tokens, config=self._config)

    @staticmethod
    def get_config(hidden_size: int, num_query_tokens: int):
        """
        Get model config

        Args:
            hidden_size:
            num_query_tokens:

        Returns:

        """
        config = AutoConfig.from_pretrained('microsoft/biogpt')
        config.encoder_width = hidden_size
        config.hidden_size = hidden_size
        # Insert cross-attention layer every other block
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        config.query_length = num_query_tokens
        return config

    @staticmethod
    def get_query_tokens(num_query_tokens: int, config):
        """
        Get query tokens

        Args:
            num_query_tokens:
            config:

        Returns:

        """
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=config.initializer_range)
        return query_tokens

    @staticmethod
    def get_model(config):
        """
        Get the QFormer model
        Args:
            config:

        Returns:

        """
        q_former = BertLMHeadModel(config=config)
        q_former.cls = None
        q_former.bert.embeddings.word_embeddings = None
        q_former.bert.embeddings.position_embeddings = None
        for layer in q_former.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        return q_former


class CNNEncoder(nn.Module):
    """
    CNN layer to encode the ECG/Blood to pass to the vision transformer
    """
    def __init__(
            self,
            image_input_type: str,
            image_size: int,
            patch_size: int,
            hidden_size: int,
            in_channels: int,
            normalization: Optional[int],
            initializer_range: float
    ):
        super().__init__()
        self._initializer_range = initializer_range
        if image_input_type == 'ecg':
            self._encoder = ECGCNNEncoder(
                patch_size=patch_size,
                hidden_size=hidden_size,
                in_channels=in_channels,
                normalization=normalization
            )
        elif image_input_type == 'cyto':
            self._encoder = CytoCNNEncoder(
                patch_size=patch_size,
                hidden_size=hidden_size,
                in_channels=in_channels,
                normalization=normalization
            )
        else:
            raise NotImplementedError(f'CNN Encoder for {image_input_type} not implemented')
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self._initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_pre_cnn_norm_layer(self):
        """

        Returns:

        """
        return self._encoder.get_pre_cnn_norm_layer()

    def forward(self, image):
        """
        Pass image through the CNN

        Args:
            image:

        Returns:

        """
        return self._encoder.forward(image)


class ECGCNNEncoder(nn.Module):
    """
    CNN layer to encode the ECG to pass to the vision transformer
    """
    def __init__(
            self,
            patch_size: int,
            hidden_size: int,
            in_channels: int,
            normalization: Optional[int]
    ):
        super().__init__()
        if normalization is None:
            self._norm = nn.Identity()
        else:
            self._norm = nn.GroupNorm(num_groups=normalization, num_channels=in_channels)
        # self.conv1 = nn.Conv1d(
        #     in_channels=in_channels,
        #     out_channels=hidden_size,
        #     kernel_size=patch_size,
        #     stride=patch_size,
        #     bias=False
        # )
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=(in_channels, patch_size),
            stride=patch_size,
            bias=False
        )

    def get_pre_cnn_norm_layer(self):
        """

        Returns:

        """
        return self._norm

    def forward(self, image):
        """
        Pass the ECG sample through the defined CNN architecture

        Args:
            image:

        Returns:

        """
        # The input is of shape (batch_size, leads, time)
        # The result of the convolution is (batch_size, out_channels, seq_len)
        # where out_channels represents the number of filters used and seq_len is the
        # resulting length after applying a single convolution filter and striding along
        # the image. Since the HF transformer expects input as (batch_size, seq_len, hidden_size)
        # we permute the dimensions
        # If using 1D Convolution
        # return self.conv1(self._norm(image)).permute(0, 2, 1)
        # If using 2D Convolution
        return self.conv1(self._norm(image).unsqueeze(1)).squeeze(2).permute(0, 2, 1)


class CytoCNNEncoder(nn.Module):
    """
    CNN layer to encode the ECG to pass to the vision transformer
    """
    def __init__(
            self,
            patch_size: int,
            hidden_size: int,
            in_channels: int,
            normalization: Optional[int]
    ):
        super().__init__()

        if normalization is None:
            self._norm = nn.Identity()
        else:
            self._norm = nn.GroupNorm(num_groups=normalization, num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

    def get_pre_cnn_norm_layer(self):
        """

        Returns:

        """
        return self._norm

    def forward(self, image):
        """
        Pass the Cyto sample through the defined CNN architecture

        Args:
            image:

        Returns:

        """
        # The input is of shape (batch_size, grid, grid, channels)
        image = self.conv1(self._norm(image))
        # Output from above is (batch_size, hidden_size, grid, grid)
        # We now flatten this
        image = image.reshape(image.shape[0], image.shape[1], -1)
        # We now end up with (batch_size, hidden_size, grid * grid)
        # Since the HF transformer expects input as (batch_size, seq_len, hidden_size)
        # we permute the dimensions
        return image.permute(0, 2, 1)


class VisionEncoder(nn.Module):
    """
    Encode the ECG using a transformer architecture
    """

    def __init__(
            self,
            image_input_type: str,
            image_size: int,
            patch_size: int,
            config: Union[AutoConfig, BioGptConfig],
            transformer: Union[VisionBioGPTModel, MaskedVisionBioGPTModel],
            in_channels: int,
            normalization: Optional[int]
    ):
        # TODO: Add init parameters?
        super().__init__()
        self._config = config
        self._transformer = transformer
        self._hidden_size = self.get_hidden_size()
        self._scale = self.get_text_embedding_scale()

        self._cnn_encoder = CNNEncoder(
            image_input_type=image_input_type,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=self._hidden_size,
            in_channels=in_channels,
            normalization=normalization,
            initializer_range=self._config.initializer_range
        )

    def get_pre_cnn_norm_layer(self):
        """

        Returns:

        """
        return self._cnn_encoder.get_pre_cnn_norm_layer()

    def get_hidden_size(self):
        """
        Get the embedding dimension

        Returns:

        """
        return self._config.hidden_size

    def get_text_embedding_scale(self):
        """
        Returns the embedding scale used in the text model

        Returns:

        """
        return self._transformer.get_embedding_scale()

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """
        Gradient checkpointing

        Args:
            enable:

        Returns:

        """
        self.transformer.grad_checkpointing = enable

    def get_input_embeddings(self, image):
        """
        Given the input ECG - convert it into an embedding that can be used
        by the transformer. For example, we can use a 1-D CNN to process
        the input and the output is passed to the transformer

        Args:
            image:

        Returns:

        """
        input_embeddings = self._cnn_encoder(image) * self._scale
        return input_embeddings

    def forward(
            self,
            image: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        """
        Model.forward

        Args:
            image:

        Returns:

        """

        # Output of say a 1-D CNN - representation that is appropriate for transformers
        input_embeddings = self.get_input_embeddings(image=image)
        outputs = self._transformer(inputs_embeds=input_embeddings)
        return outputs


class MultimodalDecoder(nn.Module):
    """
    Combine the vision encoder and text decoder
    """
    def __init__(
            self,
            vision_encoder: VisionEncoder,
            text_decoder: Union[AutoModel, BioGptModel],
            q_former: QFormer,
            projection_type: str,
            ignore_index: int,
    ):

        super().__init__()
        # TODO: Support for multiple images/text - currently works with one image and one text
        self._vision_encoder = vision_encoder
        self._text_decoder = text_decoder
        self._q_former = q_former
        self._vision_layer_norm_final = self.get_vision_layer_norm(hidden_size=self._vision_encoder.get_hidden_size())
        self._text_embedding_scale = self.get_text_embedding_scale()
        self._ignore_index = ignore_index
        # TODO: Also support attention projection: with QFormer maybe we can skip implementing this
        self._projection_type = projection_type
        if self._projection_type == 'linear':
            self._projection = nn.Linear(self._vision_encoder.get_hidden_size(), self.get_hidden_size())
        else:
            raise NotImplementedError(f'{projection_type} not implemented')

    def get_hidden_size(self):
        """
        Get the embedding dimension

        Returns:

        """
        return self._text_decoder.config.hidden_size

    def get_vision_layer_norm(self, hidden_size):
        """
        Get the layer norm to use

        Args:
            hidden_size:

        Returns:

        """
        # Without Q Former we pass the latent representation from the vision model
        # directly to the projection layer. We noticed nan loss when
        # we included a layer norm
        if self._q_former is None:
            return nn.Identity()
        else:
            return nn.LayerNorm(hidden_size)

    def get_image_embeddings(self, image):
        """
        Pass the image through a vision model and return the embeddings

        Args:
            image:

        Returns:

        """
        hidden_states = self._vision_layer_norm_final(self._vision_encoder(image)[0])
        if self._q_former is not None:
            attention_mask = torch.ones(hidden_states.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self._q_former.query_tokens.expand(hidden_states.shape[0], -1, -1)
            query_output = self._q_former.model.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = query_output.last_hidden_state
        hidden_states = self._projection(hidden_states)
        return hidden_states * self._text_embedding_scale

    def get_token_embeddings(self, input_ids):
        """
        Return the token embeddings

        Args:
            input_ids:

        Returns:

        """
        if isinstance(self._text_decoder.base_model, LoraModel):
            return self._text_decoder.base_model.base_model.embed_tokens(input_ids)
        else:
            return self._text_decoder.base_model.embed_tokens(input_ids)

    def get_text_embedding_scale(self):
        """
        Returns the embedding scale used in the text model
        Returns:

        """
        if isinstance(self._text_decoder.base_model, LoraModel):
            return self._text_decoder.base_model.base_model.embed_tokens.embed_scale
        else:
            return self._text_decoder.base_model.embed_tokens.embed_scale

    def get_text_config(self):
        """
        Return the config object of the text model

        Returns:

        """
        return self._text_decoder.config

    def get_multi_modal_embeddings(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Get the image and text input embeddings

        Args:
            images:
            input_ids:

        Returns:

        """
        image_embeddings = self.get_image_embeddings(image=images)
        token_embeddings = self.get_token_embeddings(input_ids=input_ids)
        return image_embeddings, token_embeddings

    def get_labels(self, image_embeddings: torch.FloatTensor, labels: torch.LongTensor):
        """
        Given the original labels for the text portions, we concatenate labels
        for the imag "tokens" - where we add ignore index label for the image
        tokens

        Args:
            image_embeddings:
            labels:

        Returns:

        """
        image_labels = (
            torch
            .empty(image_embeddings.shape[:-1], dtype=labels.dtype, device=labels.device)
            .fill_(self._ignore_index)
        )
        return torch.cat([image_labels, labels], dim=1)

    @staticmethod
    def concat_image_token_embeddings(
            image_embeddings: torch.FloatTensor, token_embeddings: torch.FloatTensor
    ):
        """
        Combine image and text input embeddings

        Args:
            image_embeddings:
            token_embeddings:

        Returns:

        """
        return torch.cat([image_embeddings, token_embeddings], dim=1)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            use_cache: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            weights: Optional[torch.LongTensor] = None
    ):
        """
        Get the multimodal input embeddings and pass them to the text transformer

        Args:
            input_ids:
            attention_mask:
            past_key_values:
            use_cache:
            images:
            labels:
            weights:

        Returns:

        """
        # TODO: Implement caching for faster inference
        use_cache = use_cache if use_cache is not None else self.get_text_config().use_cache
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        image_embeddings, token_embeddings = self.get_multi_modal_embeddings(
            images=images, input_ids=input_ids
        )
        multi_modal_embeddings = self.concat_image_token_embeddings(
            image_embeddings=image_embeddings, token_embeddings=token_embeddings
        )
        if labels is not None:
            multi_modal_labels = self.get_labels(image_embeddings=image_embeddings, labels=labels)
        else:
            multi_modal_labels = None
        if weights is not None:
            multi_modal_weights = self.get_labels(image_embeddings=image_embeddings, labels=weights)
        else:
            multi_modal_weights = None

        if attention_mask is None:
            attention_mask = torch.ones(
                (multi_modal_embeddings.shape[0], multi_modal_embeddings.shape[1] + past_key_values_length),
                dtype=torch.bool,
                device=multi_modal_embeddings.device,
            )
        else:
            image_attention_mask = torch.ones(
                (multi_modal_embeddings.shape[0], image_embeddings.shape[1] + past_key_values_length),
                dtype=torch.bool,
                device=multi_modal_embeddings.device,
            )
            attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        outputs = self._text_decoder(
            inputs_embeds=multi_modal_embeddings,
            attention_mask=attention_mask,
            labels=None
        )

        return outputs[0], multi_modal_labels, multi_modal_weights

    def lock_text_decoder(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        """
        Freeze text decoder

        Args:
            unlocked_layers:
            freeze_layer_norm:

        Returns:

        """
        if not unlocked_layers:  # full freezing
            print('Text decoder frozen')
            for n, p in self._text_decoder.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return
        else:
            raise NotImplementedError()


# TODO check if CLS/register tokens improve reconstruction - One of the papers I read said it doesn't help, but we
#  can still try if required
class MaskedAutoencoderVisionEncoder(nn.Module):
    """
    Encode the ECG using a transformer architecture
    """

    def __init__(
            self,
            encoder: VisionEncoder,
            decoder: VisionBioGPTModel,
            image_input_type: str,
            patch_size: int,
            in_channels: int,
    ):
        super(MaskedAutoencoderVisionEncoder, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

        # MAE decoder specifics
        self._decoder_embedding = nn.Linear(self._encoder.get_hidden_size(), self._decoder.get_hidden_size())
        self._mask_token = nn.Parameter(torch.zeros(1, 1, self._decoder.get_hidden_size()))

        # Normalization layers
        self._encoder_norm = nn.LayerNorm(self._encoder.get_hidden_size())
        self._decoder_norm = nn.LayerNorm(self._decoder.get_hidden_size())

        self._image_input_type = image_input_type
        self._patch_size = patch_size
        self._in_channels = in_channels

        # Decoder to patch
        if self._image_input_type == 'ecg':
            self._decoder_prediction = nn.Linear(
                self._decoder.get_hidden_size(),
                patch_size * in_channels,
                bias=True
            )
        elif self._image_input_type == 'cyto':
            self._decoder_prediction = nn.Linear(
                self._decoder.get_hidden_size(),
                (patch_size ** 2) * in_channels,
                bias=True
            )
        else:
            raise ValueError(f'Invalid image input type: {self._image_input_type}')

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encoder

        Args:
            x:

        Returns:

        """
        outputs = self._encoder(x)
        hidden_states, mask, ids_restore = outputs[0], outputs[1], outputs[2]
        # Normalize x - because we removed final layer norm in vision bio gpt model
        hidden_states = self._encoder_norm(hidden_states)
        return hidden_states, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Decoder

        Args:
            x:
            ids_restore:

        Returns:

        """

        # Embed tokens - pass output of encoder through a linear projection
        x = self._decoder_embedding(x)
        # Append mask tokens to sequence
        mask_tokens = self._mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # Insert mask tokens and re-order
        x_ = torch.cat([x, mask_tokens], dim=1)
        # ids_restore can get back the original ordering
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # Get the outputs from the decoder
        outputs = self._decoder(inputs_embeds=x)
        # Normalize x - because we removed final layer norm in vision bio gpt model
        hidden_states = self._decoder_norm(outputs[0])
        hidden_states = self._decoder_prediction(hidden_states)
        return hidden_states

    def patchify(self, images, normalize_labels):
        """
        Get the labels

        Args:
            images:
            normalize_labels:

        Returns:

        """
        if self._image_input_type == 'ecg':
            return self.patchify_ecg(images=images, normalize_labels=normalize_labels)
        elif self._image_input_type == 'cyto':
            return self.patchify_cyto(images=images, normalize_labels=normalize_labels)
        else:
            raise ValueError(f'Invalid image input type: {self._image_input_type}')

    def patchify_ecg(self, images, normalize_labels):
        """
        Get the labels for ECG
        """
        if normalize_labels:
            norm_layer = self._encoder.get_pre_cnn_norm_layer()
            images = norm_layer(images)

        x = images.reshape(
            shape=(
                images.shape[0],
                images.shape[2] // self._patch_size,
                self._patch_size * self._in_channels
            )
        )
        return x


    def unpatchify_ecg(self, x):
        """
        Reverse the patchify operation for ECG
        """
        # TODO normalization not reversible
        # Reshape x back to original image dimensions
        images = x.reshape(
            shape=(
                x.shape[0],
                self._in_channels,
                x.shape[1] * self._patch_size
            )
        )

        return images

    def patchify_cyto(self, images, normalize_labels):
        """
        images: (N, 6, H, W)
        x: (N, L, patch_size**2 *6)
        """

        if normalize_labels:
            norm_layer = self._encoder.get_pre_cnn_norm_layer()
            images = norm_layer(images)

        assert images.shape[2] == images.shape[3] and images.shape[2] % self._patch_size == 0, f'Shape: {images.shape}'

        height, width = images.shape[2] // self._patch_size, images.shape[2] // self._patch_size
        x = images.reshape(
            shape=(
                images.shape[0],
                height,
                self._patch_size,
                width,
                self._patch_size,
                self._in_channels,
            )
        )
        # Basically a re-shape
        x = torch.einsum('nhpwqc->nhwpqc', x)
        x = x.reshape(shape=(images.shape[0], height * width, (self._patch_size ** 2) * self._in_channels))
        return x

    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1]**.5)
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs
