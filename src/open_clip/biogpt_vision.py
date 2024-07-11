"""Bio gpt models for vision - slightly modifies architecture - custom attention mask"""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers import (
    BioGptModel,
    BioGptConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


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

    def get_hidden_size(self):
        """
        Get the embedding dimension

        Returns:

        """
        return self.config.hidden_size

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

    def get_hidden_states(
            self,
            input_ids: Optional[torch.LongTensor],
            inputs_embeds: Optional[torch.FloatTensor],
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]],
            default_attention_mask: Optional[torch.FloatTensor],
    ) -> torch.Tensor:
        """
        Same function as defined in BioGptModel with a modification to the
        attention mask - we replace the causal mask with bidirectional mask
        so that the vision tokens can attend in both directions.

        Args:
            input_ids:
            inputs_embeds:
            past_key_values:
            default_attention_mask:

        Returns:

        """
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

        # Embed positions
        # TODO: Add PatchDropout?
        positions = self.embed_positions(default_attention_mask, past_key_values_length)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        return hidden_states

    def layer_forward(
            self,
            hidden_states: Optional[torch.LongTensor],
            attention_mask: Optional[torch.FloatTensor],
            head_mask: Optional[torch.FloatTensor],
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]],
            use_cache: Optional[bool],
            output_attentions: Optional[bool],
            output_hidden_states: Optional[bool],
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Forward function through all the transformer layers

        Args:
            hidden_states:
            attention_mask:
            head_mask:
            past_key_values:
            use_cache:
            output_attentions:
            output_hidden_states:

        Returns:

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

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
        # maybe these values along with a projection and layer might
        # lead to small values when passed to the text decoder
        # hidden_states = self.layer_norm(hidden_states)

        next_cache = next_decoder_cache if use_cache else None

        return hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions

    def get_output(
            self,
            hidden_states: Optional[torch.LongTensor],
            next_cache,
            all_hidden_states,
            all_self_attns,
            all_cross_attentions,
            return_dict
    ):
        """
        Return transformer output in specified format
        Args:
            hidden_states:
            next_cache:
            all_hidden_states:
            all_self_attns:
            all_cross_attentions:
            return_dict:

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
        attention mask - we replace the causal mask with bidirectional mask
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

        hidden_states = self.get_hidden_states(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            default_attention_mask=default_attention_mask,

        )

        if attention_mask is None:
            batch_size, seq_len, _ = hidden_states.shape
            attention_mask = self.get_attention_mask(
                batch_size=batch_size,
                sequence_length=seq_len,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )

        hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions = self.layer_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return self.get_output(
            hidden_states=hidden_states,
            next_cache=next_cache,
            all_hidden_states=all_hidden_states,
            all_self_attns=all_self_attns,
            all_cross_attentions=all_cross_attentions,
            return_dict=return_dict
        )


class MaskedVisionBioGPTModel(VisionBioGPTModel):
    """
    Modify the BioGPT model to support bidirectional (full) attention mask
    """

    def __init__(self, config: BioGptConfig, mask_ratio):
        super().__init__(config)
        self._mask_ratio = mask_ratio

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masked auto encoder masking

        Args:
            x:

        Returns:

        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self._mask_ratio))

        # Create a tensor with some noise
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # Sort noise for each sample
        # Arg-sort returns the indices that sort a tensor along a given dimension in ascending
        # order by value. If the result is [14, 8, 0, ...] it means that if we place the 14th element
        # in the noise array first, the 8 the element second and the zeroth element first and so on
        # the resulting array will be in sorted order.
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is to remove
        # The result of this is say [8, 15, ..]. This means that the 8th value
        # in ids_shuffle corresponds to the zeroth value in noise, and the 15th value
        # corresponds to the 1st value in noise - so we can use this to get the
        # initial ordering back
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        # We subset x - that only the indices corresponding to ids_shuffle remain
        # This subset will be passed to the model. We basically end up using the 5 positions
        # where the noise value was the least - because of the arg-sort. The first "len_keep" elements
        # in ids_shuffle represent the indexes of the noise elements with the least amount of
        # noise
        ids_keep = ids_shuffle[:, :len_keep]
        # torch.gather creates a new tensor from the input tensor by taking the values from each
        # row along the input dimension dim. The values in torch.LongTensor, passed as index,
        # specify which value to take from each 'row'. The dimension of the output tensor is
        # same as the dimension of index tensor.
        # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
        # dim = 1 - gather from the length dimension. Assume index is [14, 8, 0, ...]. The first element
        # will be the 14th element of x, the second will be the 8th and so on
        # The order of x_masked gets jumbled up - but doesn't matter since we already added
        # position embeddings, and we're using bidirectional attention masks - so it
        # should not make a difference
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is to remove
        # 1 represent the positions that will be masked - the ones we will make
        # predictions on
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Un shuffle to get the binary mask
        # This will move the one indexes around the right spots
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def get_output(
            self,
            hidden_states: Optional[torch.LongTensor],
            next_cache,
            all_hidden_states,
            all_self_attns,
            all_cross_attentions,
            return_dict,
            mask=None,
            ids_restore=None
    ):
        """
        Return transformer output in specified format
        Args:
            hidden_states:
            next_cache:
            all_hidden_states:
            all_self_attns:
            all_cross_attentions:
            return_dict:
            mask:
            ids_restore:

        Returns:

        """
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    mask,
                    ids_restore,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions
                ]
                if v is not None
            )
        else:
            raise NotImplementedError('Need to implement a custom BaseModelOutputWithPastAndCrossAttentions class')

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
        attention mask - we replace the causal mask with bidirectional mask
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

        hidden_states = self.get_hidden_states(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            default_attention_mask=default_attention_mask

        )
        hidden_states, mask, ids_restore = self.random_masking(hidden_states)

        if attention_mask is None:
            batch_size, seq_len, _ = hidden_states.shape
            attention_mask = self.get_attention_mask(
                batch_size=batch_size,
                sequence_length=seq_len,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )

        hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions = self.layer_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return self.get_output(
            hidden_states=hidden_states,
            next_cache=next_cache,
            all_hidden_states=all_hidden_states,
            all_self_attns=all_self_attns,
            all_cross_attentions=all_cross_attentions,
            return_dict=return_dict,
            mask=mask,
            ids_restore=ids_restore
        )
