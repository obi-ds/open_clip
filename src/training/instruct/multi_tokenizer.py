"""Define an interface for the generative text tasks"""
import torch
from typing import List

from .multi_instruct_tokenizer import MultiInstructTokenizer

class MultiTokenizer(MultiInstructTokenizer):
    """
    Define functions that are used to build, tokenize and return input
    ids and labels that will be used by the model for generative tasks
    The input, output and padding. Autoregressive LM - we give it an input and train on
    predicting the next word.
    """

    def __init__(self, tokenizer, max_seq_length, pad_id):
        """
        Initialize variables
        Args:
            tokenizer (): The tokenizer used for training
            max_seq_length (int): The maximum sequence length allowed by model/tokenizer
            pad_id (int): The id used for padding tokens
        """
        super().__init__(tokenizer, max_seq_length, pad_id)

    def get_tokens(self, multi_task_instructions, return_tensor: bool = True):
        all_input_ids = []
        for instruction_input, instruction_output in multi_task_instructions:
            # Get the tokens for the instruction input - The eos and padding tokens are removed
            input_tokens = self.get_input_tokens(input_text=instruction_input)
            # Get the tokens for the instruction target - the bos and padding tokens are removed
            output_tokens = self.get_output_tokens(output_text=instruction_output)

            # The input to the model - will consist of the instruction input tokens,
            # instruction target tokens and padding tokens. Target tokens are included in
            # autoregressive lm - we specify the labels in the next step (next token prediction)
            input_ids = self.get_input_ids(input_tokens=input_tokens, output_tokens=output_tokens)

            # Concatenate inputs ids and labels from the different instructions
            all_input_ids.extend(input_ids)

        # Truncate tokens, add sentence boundaries and then pad tokens
        all_input_ids = self.pad_tokens(
            tokens=self.add_input_ids_boundaries(
                input_ids=self.truncate_tokens(tokens=all_input_ids)
            )
        )

        all_labels = self.add_labels_boundaries(labels=all_input_ids[1:])

        if return_tensor:
            return torch.tensor(all_input_ids), torch.tensor(all_labels)
        else:
            return all_input_ids, all_input_ids

    def add_input_ids_boundaries(self, input_ids):
        return input_ids + [self._tokenizer.eot_token_id]

    def add_labels_boundaries(self, labels):
        return labels + [self._pad_id]

    def pad_tokens(self, tokens) -> List[int]:
        """
        Return a tensor that contains the padding id. This will be used
        to pad the final instruction (that contains instruction input and target)
        Args:

        Returns:
            (List[int]): Tensor containing the padding tokens
        """
        return tokens + [self._pad_id] * (self._max_seq_length - len(tokens))

    def truncate_tokens(self, tokens) -> List[int]:
        # -1 - for the eod token we will add at the end
        if len(tokens) > self._max_seq_length - 1:
            tokens = tokens[:self._max_seq_length]
        return tokens
