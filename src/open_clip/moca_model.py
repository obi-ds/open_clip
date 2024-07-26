"""Moca model"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from transformers import AutoConfig, AutoModelForCausalLM

from .model import MocaVisionEncoderConfig, MocaTextDecoderConfig
from .biogpt_vision import VisionBioGPTModel
from .transformer_decoder import VisionEncoder, MultimodalDecoder, QFormer

from transformers import (
    LogitsProcessorList,
    TopPLogitsWarper,
    TopKLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
    MaxLengthCriteria,
    StoppingCriteriaList
)
from peft import get_peft_model, LoraConfig

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

        print('Vision Config: ', vision_cfg)
        print('Text Config: ', text_cfg)

        visual = self.get_vision_encoder(
            image_input_type=vision_cfg.image_input_type,
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            in_channels=vision_cfg.in_channels,
            model_name_or_path=vision_cfg.hf_model_name,
            normalization=vision_cfg.normalization,
            pretrained=vision_cfg.pretrained,
            lora=vision_cfg.lora
        )

        text = self.get_text_decoder(
            model_name_or_path=text_cfg.hf_model_name,
            pretrained=text_cfg.pretrained,
            lora=text_cfg.lora
        )

        if vision_cfg.q_former:
            print('Using Q Former')
            q_former = self.get_q_former(hidden_size=visual.get_hidden_size(), num_query_tokens=vision_cfg.q_former)
        else:
            q_former = None

        self._pad_token_id = self.get_pad_token_id(text)

        self._multimodal_decoder = MultimodalDecoder(
            vision_encoder=visual,
            text_decoder=text,
            q_former=q_former,
            projection_type=text_cfg.projection_type,
            ignore_index=ignore_index,
        )

        # Set these to None - so that it works with the existing open_clip implementation
        self.visual = None
        self.text = None

    def get_vision_encoder(
            self,
            image_input_type: str,
            image_size: int,
            patch_size: int,
            model_name_or_path: str,
            in_channels: int,
            normalization: Optional[int],
            pretrained: Optional[str],
            lora: bool
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
            pretrained:
            lora:

        Returns:

        """
        # TODO: Currently we only support the BioGPT architecture
        #       for other architectures - we need to modify the attention mask accordingly
        #       and create another subclass
        config = AutoConfig.from_pretrained(model_name_or_path)
        transformer = VisionBioGPTModel(config=config)

        vision_encoder = VisionEncoder(
            image_input_type=image_input_type,
            image_size=image_size,
            patch_size=patch_size,
            config=config,
            transformer=transformer,
            in_channels=in_channels,
            normalization=normalization
        )
        if pretrained:
            print(f'Loading pre-trained vision encoder model: ', pretrained)
            vision_encoder.load_state_dict(torch.load(pretrained))
        if lora:
            print('Adding LoRA adapters - Vision Encoder')
            vision_encoder._transformer = self.get_lora_model(model=vision_encoder._transformer)
        return vision_encoder

    def get_text_decoder(self, model_name_or_path, pretrained, lora: bool):
        """
        Return the text decoder model (from huggingface)

        Args:
            model_name_or_path:
            pretrained:
            lora:

        Returns:

        """
        config = AutoConfig.from_pretrained(model_name_or_path)
        if not pretrained:
            model = AutoModelForCausalLM.from_config(
                config=config,
            )
        else:
            print(f'Loading pre-trained text causal lm model: ', pretrained)
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
            )
        if lora:
            print('Adding LoRA adapters - Text Causal LM')
            model = self.get_lora_model(model=model)
        return model

    @staticmethod
    def get_q_former(hidden_size: int, num_query_tokens: int):
        """
        Return the QFormer model object
        Args:
            hidden_size:
            num_query_tokens:

        Returns:

        """
        print('Number of query tokens: ', num_query_tokens)
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
        """
        Lock text tower
        Args:
            unlocked_layers:
            freeze_layer_norm:

        Returns:

        """
        self._multimodal_decoder.lock_text_decoder(unlocked_layers, freeze_layer_norm)

    @staticmethod
    def get_lora_model(model, rank=2, lora_alpha=16, lora_dropout=0.1, target_modules=None):
        """
        Add LORA adapters to the model
        Args:
            model:
            rank:
            lora_alpha:
            lora_dropout:
            target_modules:

        Returns:

        """
        # TODO: Specify task type? - will it work without?
        if target_modules is None:
            target_modules = ['k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2']

        peft_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias='none',
            target_modules=target_modules
        )
        model = get_peft_model(model, peft_config)
        print(model.print_trainable_parameters())
        return model

    def forward(
            self,
            images,
            texts,
    ):

        if texts.dim() == 3:
            if texts.shape[1] == 2:
                labels = texts[:, 1, :]
                input_ids = texts[:, 0, :]
                weights = None
            else:
                labels = texts[:, 1, :].long()
                input_ids = texts[:, 0, :].long()
                weights = texts[:, 2, :]
        else:
            labels = None
            weights = None
            input_ids = texts

        # 0 for pad token positions
        attention_mask = (
            (input_ids != self._pad_token_id)
            .to(dtype=torch.bool, device=input_ids.device)
        )

        multimodal_logits, multimodal_labels, multi_modal_weights = self._multimodal_decoder(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            labels=labels,
            weights=weights
        )

        return {
            "logits": multimodal_logits,
            "labels": multimodal_labels,
            "weights": multi_modal_weights
        }

    def generate(
            self,
            image,
            text=None,
            seq_len=30,
            max_seq_len=77,
            temperature=1.,
            generation_type=None,
            top_p=0.1,  # keep tokens in the 1 - top_p quantile
            top_k=1,  # keeps the top_k most probable tokens
            pad_token_id=None,
            eos_token_id=None,
            sot_token_id=None,
            min_seq_len=5,
            stopping_criteria=None,
            repetition_penalty=1.0,
            fixed_output_length=False  # if True output.shape == (batch_size, seq_len)
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
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

            if generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

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

                    if cur_len + 1 == seq_len:
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if all(stopping_criteria(out, None)):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out


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
