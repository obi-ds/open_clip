"""Define an interface for the instruction tasks"""
from typing import Tuple, List, Union

import torch


class MultiInstructInterface(object):
    """
    An interface for instruction tasks. Define functions that are used
    to build, tokenize and return instructions. The input, output and padding
    tokens are processed separately. This is because we can create the labels
    and label mask based on the output tokens and also create the input_ids
    based on all of them. Autoregressive LM - we give it an input and train on
    predicting the next word. Here we only want to train on the instruction target
    tokens. It's easier to build a label tensor/mask for this if we process everything
    separately
    TODO - Handle truncation/padding when instruction token length exceeds max seq length
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
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad_id = pad_id

    def process_instructions(
            self,
            instructions: List[Tuple[str, str]]
    ) -> Tuple[str, str]:
        """
        Join the different instruction inputs into one single string. Keep the target
        of the final instruction (the instruction we train on) separate. This step
        should result in two strings, one that contains the input instruction (this
        can contain the task definition, inputs and targets of the k_shot examples),
        the other string is the instruction target
        Args:
            instructions (List[Tuple[str, str]]): A list that contains tuples which have the instruction input and
            target. The list will contain only 1 element for zero shot training

        Returns:
            (Tuple[str, str]): Tuple that contains the instruction input text concatenated across the k-shot examples
            and the instruction target text
        """
        input_text = ''
        output_text = ''
        k_shot = len(instructions) - 1
        for index, instruction in enumerate(instructions):
            if index != k_shot:
                input_text += ''.join(instruction)
            else:
                input_text += instruction[0]
                output_text = instruction[1]
        return input_text, output_text

    def tokenize_instruction(
            self,
            input_text: str,
            output_text: str
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[str, str]]:
        """
        Given the instruction input and instruction target/output texts.
        Tokenize the texts to return the input_ids that will be passed to the
        model and the labels that will be used to train/evaluate the model
        Args:
            input_text (str): Instruction input
            output_text (str): Instruction target

        Returns:
            (Union[Tuple[torch.Tensor, torch.Tensor], Tuple[str, str]]): String representations of the instruction
            inputs and targets are returned if tokenizer is None, else the tokenized representations of these
            are returned
        """
        # If tokenizer is None, return the texts as is.
        if self._tokenizer is None:
            return input_text, output_text

        # Get the tokens for the instruction input - The eos and padding tokens are removed
        input_tokens = self.get_input_tokens(input_text=input_text)
        # Get the tokens for the instruction target - the bos and padding tokens are removed
        output_tokens = self.get_output_tokens(output_text=output_text)
        padding_length = self._max_seq_length - len(input_tokens) - len(output_tokens)
        # Get the padding tokens - to make up the max seq length
        padding_tokens = self.get_padding_tokens(padding_length=padding_length)
        # The input to the model - will consist of the instruction input tokens,
        # instruction target tokens and padding tokens. Target tokens are included in
        # autoregressive lm - we specify the labels in the next step (next token prediction)
        input_ids = self.get_input_ids(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            padding_tokens=padding_tokens
        ).unsqueeze(axis=0)
        # Get the tensor that will contain the labels for autoregressive training, such that we
        # only train on the instruction targets
        labels = self.get_labels(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            padding_tokens=padding_tokens
        ).unsqueeze(axis=0)
        return torch.cat([input_ids, labels])

    def get_input_tokens(self, input_text: str) -> List[int]:
        """
        Tokenize the input instruction and return the tokens
        Returns all tokens but the eos token - since we eventually
        add the instruction target to this
        Args:
            input_text (str): Any input text

        Returns:
            (torch.Tensor): Return tokenized input
        """
        return self._tokenizer.encode(input_text)

    def get_output_tokens(self, output_text: str) -> List[int]:
        """
        Tokenize the input instruction and return the tokens
        Returns all tokens but the bos token - since we eventually
        add the instruction input to this
        Args:
            output_text (str): Any input text

        Returns:
            (List[int]): Return tokenized input
        """
        return self._tokenizer.encode(output_text)

    def get_input_ids(
            self,
            input_tokens: List[int],
            output_tokens: List[int],
    ) -> List[int]:
        """
        Given the instruction input tokens, instruction target tokens and padding tokens,
        concatenate them and return one tensor that will be passed to the model
        Args:
            input_tokens (List[int]): The instruction input tokens
            output_tokens (List[int]): The instruction output tokens

        Returns:
            (List[int]): The tokens that are inputs to the model
        """
        return input_tokens + output_tokens

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
        if self._pad_id == 0:
            return [0] * (len(input_tokens) - 1) + output_tokens + [self._tokenizer.eot_token_id]
        else:
            raise NotImplementedError()

    def add_input_ids_boundaries(self, input_ids):
        return [self._tokenizer.sot_token_id] + input_ids + [self._tokenizer.eot_token_id]

    def add_labels_boundaries(self, labels):
        return [self._pad_id] + labels + [self._pad_id]

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
        # We use minus 2 because we add the bos and eos tokens after truncating
        if len(tokens) > self._max_seq_length - 2:
            tokens = tokens[:self._max_seq_length]
        return tokens
