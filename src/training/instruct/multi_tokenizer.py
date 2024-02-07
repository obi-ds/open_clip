"""Define an interface for the generative text tasks"""
import torch
from typing import List, Sequence


class MultiTokenizer(object):
    """
    Define functions that are used to build, tokenize and return input
    ids and labels that will be used by the model for generative tasks
    The input, output and padding. Autoregressive LM - we give it an input and train on
    predicting the next word.
    """

    def __init__(self, tokenizer, max_seq_length, pad_id, ignore_index=-100):
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
        self._ignore_index = ignore_index

    def get_tokens(self, multi_task_instructions, return_tensor: bool = True):
        """
        Given a list of inputs - return the tokenizer version using the tokenizer

        Args:
            multi_task_instructions:
            return_tensor:

        Returns:

        """
        all_input_ids = []
        for instruction_input, instruction_output, _ in multi_task_instructions:
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

        # Truncate tokens, add sentence boundaries
        all_input_ids = self.add_input_ids_boundaries(
                input_ids=self.truncate_tokens(tokens=all_input_ids)
        )
        all_labels = self.add_labels_boundaries(labels=all_input_ids[1:])

        # Pad tokens
        all_input_ids = self.pad_input_tokens(all_input_ids)
        all_labels = self.pad_label_tokens(tokens=all_labels)

        if return_tensor:
            return torch.tensor(all_input_ids), torch.tensor(all_labels)
        else:
            return all_input_ids, all_labels

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

    def add_input_ids_boundaries(self, input_ids: List[int]):
        """
        Add the sot and eot tokens to the input

        Args:
            input_ids (Sequence[int]): The inputs ids given by the tokenizer

        Returns:
            (List[int]): Input ids with the eot token added
        """
        return input_ids + [self._tokenizer.eot_token_id]

    def add_labels_boundaries(self, labels: List[int]) -> List[int]:
        """
        Add the pad/ignore-index tokens to the labels

        Args:
            labels (Sequence[int]): The labels for the model

        Returns:
            (List[int]): Labels with pad/ignore-index token added
        """
        return labels + [self._ignore_index]

    def pad_tokens_with_id(self, tokens: List[int], pad_id) -> List[int]:
        """
        Return a tensor that contains the padding id. This will be used
        to pad the final instruction (that contains instruction input and target)

        Args:
            tokens (List[int]): The input tokens
            pad_id (int): The ID to use for padding

        Returns:
            (List[int]): Tensor containing the padding tokens
        """
        return tokens + [pad_id] * (self._max_seq_length - len(tokens))

    def pad_input_tokens(self, tokens: List[int]) -> List[int]:
        """
        Return a tensor that contains the padding id. This will be used
        to pad the final instruction (that contains instruction input and target)

        Args:
            tokens (List[int]): The input tokens

        Returns:
            (List[int]): Tensor containing the padding tokens
        """
        return self.pad_tokens_with_id(tokens=tokens, pad_id=self._pad_id)

    def pad_label_tokens(self, tokens: List[int]) -> List[int]:
        """
        Return a tensor that contains the padding id. This will be used
        to pad the final instruction (that contains instruction input and target)

        Args:
            tokens (List[int]): The input tokens

        Returns:
            (List[int]): Tensor containing the padding tokens
        """
        return self.pad_tokens_with_id(tokens=tokens, pad_id=self._ignore_index)

    def truncate_tokens(self, tokens: List[int]) -> List[int]:
        """
        Truncate tokens to max length

        Args:
            tokens (List[int]): The input tokens

        Returns:
            tokens (List[int]): The tokens truncated to max length if exceeded max length
        """
        # -2 - for the sot and eot tokens we will add at the end
        if len(tokens) > self._max_seq_length - 1:
            tokens = tokens[:self._max_seq_length - 1]
        return tokens

class MultiTokenizerEval(MultiTokenizer):
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
        """
        Given a list of inputs - return the tokenizer version using the tokenizer

        Args:
            multi_task_instructions:
            return_tensor:

        Returns:

        """
        all_input_ids = []
        all_labels = []
        for instruction_input, instruction_output, ignore_instruction in multi_task_instructions:
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
            labels = self.get_labels(
                input_tokens=input_tokens, output_tokens=output_tokens, ignore_instruction=ignore_instruction
            )

            # Concatenate inputs ids and labels from the different instructions
            all_input_ids.extend(input_ids)
            all_labels.extend(labels)



        # Truncate tokens, add sentence boundaries and then pad tokens
        all_input_ids = self.pad_input_tokens(
            tokens=self.add_input_ids_boundaries(
                input_ids=self.truncate_tokens(tokens=all_input_ids)
            )
        )

        # Truncate tokens, add sentence boundaries and then pad tokens
        all_labels = self.pad_input_tokens(
            tokens=self.add_labels_boundaries(labels=self.truncate_tokens(tokens=all_labels))
        )

        if return_tensor:
            return torch.tensor(all_input_ids), torch.tensor(all_labels)
        else:
            return all_input_ids, all_labels

    def get_labels(
            self,
            input_tokens: List[int],
            output_tokens: List[int],
            ignore_instruction: bool = False
    ) -> List[int]:
        """
        Given the instruction input tokens, instruction target tokens and padding tokens,
        return a tensor that contains 0 (ignore_index) values in positions that correspond
        to input and padding tokens, and output token values are left as is. We are creating
        a label mask essentially.
        Args:
            input_tokens (List[int]): The instruction input tokens
            output_tokens (List[int]): The instruction output tokens
            ignore_instruction (bool, defaults to `False`): Boolean whether to ignore the output of this instruction

        Returns:
            (List[int]): The label ids the model will be trained on
        """
        if ignore_instruction:
            return [self._ignore_index] * (len(input_tokens) + len(output_tokens))
        else:
            # return self.get_input_ids(input_tokens=input_tokens, output_tokens=output_tokens)
            # TODO: Temporary fix for evaluation - pad id is to ignore the token that starts the sentence
            #  Example: * Sprains and ... : -> ignore the probability of the "*" token and ':' token
            labels = self.get_input_ids(input_tokens=input_tokens, output_tokens=output_tokens)
            return [self._ignore_index] + labels[1:-1] + [self._ignore_index]
