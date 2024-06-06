"""Moca model"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from transformers import AutoConfig, AutoModelForCausalLM

from .model import MocaVisionEncoderConfig, MocaTextDecoderConfig
from .transformer_decoder import VisionEncoder, MultimodalDecoder, QFormer

from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    TopPLogitsWarper,
    TopKLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
    MaxLengthCriteria,
    StoppingCriteriaList
)

GENERATION_TYPES = {
    "top_k": TopKLogitsWarper,
    "top_p": TopPLogitsWarper,
    "beam_search": "beam_search"
}
_has_transformers = True


class MoCa(nn.Module):
    def __init__(
            self,
            vision_cfg: MocaVisionEncoderConfig,
            text_cfg: MocaTextDecoderConfig,
            ignore_index=-100,
            **kwargs
    ):
        print('MoCa Model')
        super().__init__()
        vision_cfg = (
            MocaVisionEncoderConfig(**vision_cfg)
            if isinstance(vision_cfg, dict) else vision_cfg
        )
        text_cfg = (
            MocaTextDecoderConfig(**text_cfg)
            if isinstance(text_cfg, dict) else text_cfg
        )

        visual = self.get_vision_encoder(
            image_input_type=vision_cfg.image_input_type,
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            in_channels=vision_cfg.in_channels,
            model_name_or_path=vision_cfg.hf_model_name,
            normalization=vision_cfg.normalization
        )

        text = self.get_text_decoder(
            model_name_or_path=text_cfg.hf_model_name,
            pretrained=text_cfg.pretrained
        )

        if vision_cfg.q_former:
            q_former = self.get_q_former(hidden_size=visual.get_hidden_size(), num_query_tokens=vision_cfg.q_former)
        else:
            q_former = None

        self._pad_token_id = self.get_pad_token_id(text)

        self._multimodal_decoder = MultimodalDecoder(
            vision_encoder=visual,
            text_decoder=text,
            q_former=q_former,
            projection_type=text_cfg.projection_type,
            ignore_index=ignore_index
        )

        # Set these to None - so that it works with the existing open_clip implementation
        self.visual = None
        self.text = None

        # The following are initialized so that the code works with existing clip code
        self.logit_scale = torch.ones(1, requires_grad=False)

    @staticmethod
    def get_vision_encoder(
            image_input_type: str,
            image_size: int,
            patch_size: int,
            model_name_or_path: str,
            in_channels: int,
            normalization: Optional[int]
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

        Returns:

        """
        return VisionEncoder(
            image_input_type=image_input_type,
            image_size=image_size,
            patch_size=patch_size,
            model_name_or_path=model_name_or_path,
            in_channels=in_channels,
            normalization=normalization
        )

    @staticmethod
    def get_text_decoder(model_name_or_path, pretrained):
        """
        Return the text decoder model (from huggingface)

        Args:
            model_name_or_path:
            pretrained:

        Returns:

        """
        config = AutoConfig.from_pretrained(model_name_or_path)
        if not pretrained:
            return AutoModelForCausalLM.from_config(
                config=config,
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
            )

    @staticmethod
    def get_q_former(hidden_size: int, num_query_tokens: int):
        """
        Return the QFormer model object
        Args:
            hidden_size:
            num_query_tokens:

        Returns:

        """
        return QFormer(
            hidden_size=hidden_size,
            num_query_tokens=num_query_tokens
        )

    @staticmethod
    def get_pad_token_id(text):
        """
        Get the id of the pad token

        Returns:

        """
        return text.config.pad_token_id

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        # FIXME: This might not work - check and fix accordingly
        self.text.lock(unlocked_layers, freeze_layer_norm)

    def forward(
            self,
            images,
            texts,
    ):

        if texts.dim() == 3:
            labels = texts[:, 1, :]
            input_ids = texts[:, 0, :]
        else:
            labels = None
            input_ids = texts

        # 0 for pad token positions
        attention_mask = (
            (input_ids != self._pad_token_id)
            .to(dtype=torch.bool, device=input_ids.device)
        )

        multimodal_logits, multimodal_labels = self._multimodal_decoder(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            "logits": multimodal_logits,
            "labels": multimodal_labels,
            "logit_scale": self.logit_scale.to(device=multimodal_logits.device)
        }


    def generate(
        self,
        image,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False # if True output.shape == (batch_size, seq_len)
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.device

            if generation_type == "beam_search":
                output = self._generate_beamsearch(
                    image_inputs=image,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    return torch.cat(
                        (output, torch.ones(output.shape[0], seq_len-output.shape[1], device=device, dtype=output.dtype) * self.pad_id),
                        dim=1
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            image_latent, image_embs = self._encode_image(image)

            if text is None:
                text = torch.ones((image.shape[0], 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            cur_len = text.shape[1]
            self.eval()
            out = text

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]
                logits = self(image, x)["logits"][:, -1]
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if (cur_len + 1 == seq_len):
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if stopping_criteria(out, None):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

    def _generate_beamsearch(
            self,
            image_inputs,
            pad_token_id=None,
            eos_token_id=None,
            sot_token_id=None,
            num_beams=6,
            num_beam_groups=3,
            min_seq_len=5,
            stopping_criteria=None,
            logit_processor=None,
            logit_warper=None,
    ):
        device = image_inputs.device
        batch_size = image_inputs.shape[0]
        image_inputs = torch.repeat_interleave(image_inputs, num_beams, dim=0)
        image_latent, image_embs = self._encode_image(image_inputs)

        input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        logits_processor = (
            LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)])
            if logit_processor is None
            else logit_processor
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, image_inputs=image_inputs)
            outputs = self(
                model_inputs['images'],
                model_inputs['text'],
                image_latent=image_latent,
                image_embs=image_embs
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs['sequences']


def prepare_inputs_for_generation(input_ids, image_inputs, past=None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = None
    return {
        "text": input_ids,
        "images": image_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
