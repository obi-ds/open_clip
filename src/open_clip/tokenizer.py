""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import os
import torch
from typing import Optional, List, Union

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HFTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int],
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.context_length = context_length

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        input_ids = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        ).input_ids

        return input_ids
