"""Model architectures for decoder only architecture"""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers import (
    AutoConfig,
    BioGptModel,
    BioGptConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


# class AttentionalPooler(nn.Module):
#     def __init__(
#             self,
#             d_model: int,
#             hidden_size: int,
#             n_head: int = 8,
#             n_queries: int = 256,
#             norm_layer: Callable = LayerNorm
#     ):
#         super().__init__()
#         self.query = nn.Parameter(torch.randn(n_queries, d_model))
#         self.attn = nn.MultiheadAttention(d_model, n_head, kdim=hidden_size, vdim=hidden_size)
#         self.ln_q = norm_layer(d_model)
#         self.ln_k = norm_layer(hidden_size)
#
#     def forward(self, x: torch.Tensor):
#         x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
#         N = x.shape[1]
#         q = self.ln_q(self.query)
#         out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]
#         return out.permute(1, 0, 2)  # LND -> NLD

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
            normalization: Optional[int]
    ):
        super().__init__()
        if image_input_type == 'ecg':
            self._encoder = ECGCNNEncoder(
                patch_size=patch_size,
                hidden_size=hidden_size,
                in_channels=in_channels,
                normalization=normalization
            )
        else:
            raise NotImplementedError(f'CNN Encoder for {image_input_type} not implemented')

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
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

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
        return self.conv1(self._norm(image)).permute(0, 2, 1)


class VisionEncoder(nn.Module):
    """
    Encode the ECG using a transformer architecture
    """

    def __init__(
            self,
            image_input_type: str,
            image_size: int,
            patch_size: int,
            model_name_or_path: str,
            in_channels: int,
            normalization: Optional[int]
    ):
        # TODO: Add init parameters?
        super().__init__()
        self._config = AutoConfig.from_pretrained(model_name_or_path)
        # TODO: Currently we only support the BioGPT architecture
        #       for other architectures - we need to modify the attention mask accordingly
        #       and create another subclass
        self._transformer = VisionBioGPTModel(config=self._config)
        self._hidden_size = self.get_hidden_size()
        self._scale = self.get_text_embedding_scale()

        self._cnn_encoder = CNNEncoder(
            image_input_type=image_input_type,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=self._hidden_size,
            in_channels=in_channels,
            normalization=normalization
        )

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
        hidden_states = self._transformer(inputs_embeds=input_embeddings)
        return hidden_states


class MultimodalDecoder(nn.Module):
    """
    Combine the vision encoder and text decoder
    """
    def __init__(
            self,
            vision_encoder,
            text_decoder,
            projection_type: str,
            ignore_index: int
    ):

        super().__init__()
        # TODO: Support for multiple images/text - currently works with one image and one text
        self._vision_encoder = vision_encoder
        self._text_decoder = text_decoder
        self._text_embedding_scale = self.get_text_embedding_scale()
        self._ignore_index = ignore_index
        # TODO: Also support attention projection
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

    def get_image_embeddings(self, image):
        """
        Pass the image through a vision model and return the embeddings

        Args:
            image:

        Returns:

        """
        hidden_states = self._vision_encoder(image)
        hidden_states = self._projection(hidden_states[0])
        return hidden_states * self._text_embedding_scale

    def get_token_embeddings(self, input_ids):
        """
        Return the token embeddings

        Args:
            input_ids:

        Returns:

        """
        return self._text_decoder.base_model.embed_tokens(input_ids)

    def get_text_embedding_scale(self):
        """
        Returns the embedding scale used in the text model
        Returns:

        """
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
            labels: Optional[torch.LongTensor] = None
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
        multi_modal_labels = self.get_labels(image_embeddings=image_embeddings, labels=labels)

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

        return outputs[0], multi_modal_labels


class VisionBioGPTModel(BioGptModel):
    """
    Modify the BioGPT model to support bidirectional (full) attention mask
    """

    def __init__(self, config: BioGptConfig):
        super().__init__(config)
        self._embed_scale = self.embed_tokens.embed_scale
        self.embed_tokens = None
        # Turning this one was causing the loss to go to NaN
        self.layer_norm = None

    @staticmethod
    def get_attention_mask(batch_size, sequence_length, device, dtype):
        """
        Build an attention mask. We want to use a bid-directional attention mask since
        we are using an encoder. We set the attention mask manually to avoid the HF transformer
        from using a causal mask. We use a full attention mask

        Returns:

        """
        attention_mask = torch.ones(
            size=(batch_size, 1, sequence_length, sequence_length), dtype=torch.long, device=device
        )
        # if the 4D mask has correct shape - invert it and fill with negative infinity
        inverted_mask = 1.0 - attention_mask
        attention_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )
        return attention_mask

    def get_embedding_scale(self):
        """
        Return the embedding scale value

        Returns:

        """
        return self._embed_scale

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            default_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Same function as defined in BioGptModel with a modification to the
        attention mask - we replace th causal mask with bidirectional mask
        so that the vision tokens can attend in both directions.

        Args:
            input_ids:
            attention_mask:
            head_mask:
            inputs_embeds:
            past_key_values:
            use_cache:
            output_attentions:
            output_hidden_states:
            return_dict:
            default_attention_mask:

        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input)

        if default_attention_mask is None:
            default_attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1] + past_key_values_length),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        elif default_attention_mask.shape[1] != past_key_values_length + input_shape[1]:
            raise ValueError(
                f"The provided attention mask has length {default_attention_mask.shape[1]}, but its length should be "
                f"{past_key_values_length + input_shape[1]} (sum of the lengths of current and past inputs)"
            )

        if attention_mask is None:
            batch_size, seq_len, _ = inputs_embeds.shape
            attention_mask = self.get_attention_mask(
                batch_size=batch_size,
                sequence_length=seq_len,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )

        # embed positions
        positions = self.embed_positions(default_attention_mask, past_key_values_length)

        hidden_states = inputs_embeds + positions

        # TODO: Add PatchDropout?
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # We removed the final layer norm - it leads to a loss with NaN
        # maybe these values along with a projection and layer and scaling might
        # lead to small values when passed to the text decoder
        # hidden_states = self.layer_norm(hidden_states)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
