"""Define an interface for the instruction tasks"""
from typing import Tuple, List, Union

import torch


class InstructInterface(object):
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
            max_seq_length: int = 77,
            pad_id: int = 0,
    ):
        """
        Initialize variables
        Args:
            tokenizer (): The tokenizer used for training
            max_seq_length (int, defaults to `77`): The maximum sequence length allowed by model/tokenizer
            pad_id (int, defaults to `0`): The id used for padding tokens
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

    def get_input_tokens(self, input_text: str) -> torch.Tensor:
        """
        Tokenize the input instruction and return the tokens
        Returns all tokens but the eos token - since we eventually
        add the instruction target to this
        Args:
            input_text (str): Any input text

        Returns:
            (torch.Tensor): Return tokenized input
        """
        input_tokens = self._tokenizer(input_text)[0]
        # Remove eos and padding tokens
        # They will be added later - after adding instruction target tokens
        return input_tokens[input_tokens != self._pad_id][:-1]

    def get_output_tokens(self, output_text: str) -> torch.Tensor:
        """
        Tokenize the input instruction and return the tokens
        Returns all tokens but the bos token - since we eventually
        add the instruction input to this
        Args:
            output_text (str): Any input text

        Returns:
            (torch.Tensor): Return tokenized input
        """
        output_tokens = self._tokenizer(output_text)[0]
        # Remove bos token and padding tokens
        # Will be added after appending input tokens to this
        return output_tokens[output_tokens != self._pad_id][1:]

    def get_padding_tokens(self, padding_length: int) -> torch.Tensor:
        """
        Return a tensor that contains the padding id. This will be used
        to pad the final instruction (that contains instruction input and target)
        Args:
            padding_length (int): The number of padding tokens required

        Returns:
            (torch.Tensor): Tensor containing the padding tokens
        """
        if self._pad_id == 0:
            return torch.zeros(padding_length, dtype=torch.long)
        else:
            raise NotImplementedError()

    def get_input_ids(
            self,
            input_tokens: torch.Tensor,
            output_tokens: torch.Tensor,
            padding_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Given the instruction input tokens, instruction target tokens and padding tokens,
        concatenate them and return one tensor that will be passed to the model
        Args:
            input_tokens (torch.Tensor): The instruction input tokens
            output_tokens (torch.Tensor): The instruction output tokens
            padding_tokens (torch.Tensor): The padding tokens

        Returns:
            (torch.Tensor): The tokens that are inputs to the model
        """
        return torch.cat([input_tokens, output_tokens, padding_tokens])

    def get_labels(
            self,
            input_tokens: torch.Tensor,
            output_tokens: torch.Tensor,
            padding_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Given the instruction input tokens, instruction target tokens and padding tokens,
        return a tensor that contains 0 (ignore_index) values in positions that correspond
        to input and padding tokens, and output token values are left as is. We are creating
        a label mask essentially.
        Args:
            input_tokens (torch.Tensor): The instruction input tokens
            output_tokens (torch.Tensor): The instruction output tokens
            padding_tokens (torch.Tensor): The padding tokens

        Returns:
            (torch.Tensor): The label ids the model will be trained on
        """
        if self._pad_id == 0:
            return torch.cat(
                [
                    torch.zeros(len(input_tokens), dtype=torch.long),
                    output_tokens,
                    padding_tokens
                ]
            )
        else:
            raise NotImplementedError()
