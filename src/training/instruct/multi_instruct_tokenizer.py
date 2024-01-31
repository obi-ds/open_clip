"""Define an interface for the instruction tasks"""
import torch
from typing import List

from .multi_tokenizer import MultiTokenizer


class MultiInstructTokenizer(MultiTokenizer):
    """
    An interface for instruction tokenizer. Define functions that are used
    to build, tokenize and return instructions. The input, output and padding
    tokens are processed separately. This is because we can create the labels
    and label mask based on the output tokens and also create the input_ids
    based on all of them. Autoregressive LM - we give it an input and train on
    predicting the next word. Here we only want to train on the instruction target
    tokens. It's easier to build a label tensor/mask for this if we process everything
    separately
    """

    def __init__(
            self,
            tokenizer,
            max_seq_length,
            pad_id,
    ):
        """
        Initialize variables
        Args:
            tokenizer (): The tokenizer used for training
            max_seq_length (int): The maximum sequence length allowed by model/tokenizer
            pad_id (int): The id used for padding tokens
        """
        super().__init__(tokenizer, max_seq_length, pad_id)

    def get_tokens(
            self,
            multi_task_instructions,
            return_tensor: bool = True,
    ):
        """
        Given a list of inputs - return the tokenizer version using the tokenizer

        Args:
            multi_task_instructions:
            return_tensor:

        Returns:

        """
        all_input_ids = []
        all_labels = []
        for instruction_input, instruction_output, _ in multi_task_instructions:
            # Get the tokens for the instruction input - The eos and padding tokens are removed
            input_tokens = self.get_input_tokens(input_text=instruction_input)
            # Get the tokens for the instruction target - the bos and padding tokens are removed
            output_tokens = self.get_output_tokens(output_text=instruction_output)

            # The input to the model - will consist of the instruction input tokens,
            # instruction target tokens and padding tokens. Target tokens are included in
            # autoregressive lm - we specify the labels in the next step (next token prediction)
            input_ids = self.get_input_ids(input_tokens=input_tokens, output_tokens=output_tokens)
            # Get the tensor that will contain the labels for autoregressive training, such that we
            # only train on the instruction targets
            labels = self.get_labels(input_tokens=input_tokens, output_tokens=output_tokens)

            # Concatenate inputs ids and labels from the different instructions
            all_input_ids.extend(input_ids)
            all_labels.extend(labels)

        # Truncate tokens, add sentence boundaries and then pad tokens
        all_input_ids = self.pad_tokens(
            tokens=self.truncate_tokens(tokens=all_input_ids)
        )
        all_labels = self.pad_tokens(
            tokens=self.truncate_tokens(tokens=all_labels)
        )

        if return_tensor:
            return torch.tensor(all_input_ids), torch.tensor(all_labels)
        else:
            return all_input_ids, all_labels

    def get_labels(
            self,
            input_tokens: List[int],
            output_tokens: List[int],
    ) -> List[int]:
        """
        Given the instruction input tokens, instruction target tokens and padding tokens,
        return a tensor that contains 0 (ignore_index) values in positions that correspond
        to input and padding tokens, and output token values are left as is. We are creating
        a label mask essentially.
        Args:
            input_tokens (List[int]): The instruction input tokens
            output_tokens (List[int]): The instruction output tokens

        Returns:
            (List[int]): The label ids the model will be trained on
        """
        return [self._pad_id] * (len(input_tokens) - 1) + output_tokens + [self._tokenizer.eot_token_id]
