"""Convert icd codes into the desired label format for training models"""
import random
from typing import Callable, List, Sequence

import numpy as np
from functools import partial


class ICDConvert(object):
    """
    Class to convert icd codes into the desired label format for training models
    """

    def __init__(
            self,
            icd_descriptions,
            billable_probability: float,
            top_non_probability: float,
            mixed_non_probability: float,
            lowercase: bool
    ):
        """
        Initialize parameters that will be used for converting icd codes
        to codes in the hierarchy and their textual form.
        Args:
            icd_descriptions (): The object to get descriptions of raw codes
            billable_probability (float): The probability of mapping the sequence into only a billable sequence. This
            would mean keeping the sequence as is
            top_non_probability (float): The probability of mapping the sequence into the top most level non-billable
            sequence of codes.
            mixed_non_probability (float): The probability of mapping the sequence into a mix of various non-billable
            and billable sequence of codes.
            lowercase (bool): Whether to lowercase the output
        """
        self._icd_descriptions = icd_descriptions
        self._billable_probability = billable_probability
        self._top_non_probability = top_non_probability
        self._mixed_non_probability = mixed_non_probability
        self.__check_probability(
            billable_probability=self._billable_probability,
            top_non_probability=self._top_non_probability,
            mixed_non_probability=self._mixed_non_probability
        )
        self._lowercase = lowercase

    def get_icd_description(self):
        """

        Returns:

        """
        return self._icd_descriptions

    @staticmethod
    def __check_probability(billable_probability, top_non_probability, mixed_non_probability):
        """
        Returns:
            (bool): True if the probabilities sum up to 1
        """
        return billable_probability + top_non_probability + mixed_non_probability != 1

    @staticmethod
    def get_billable_code(icd_code: str) -> str:
        """
        The input sequence is a sequence of billable codes, so return the code as is
        Args:
            icd_code (str): An icd code

        Returns:
            icd_code (str): The same code
        """
        return icd_code

    @staticmethod
    def get_non_billable_code(
            icd_code: str,
            topmost_level: bool = False,
            allow_billable: bool = False
    ) -> str:
        """
        Get the non-billable version of a billable code.
        Get the top level codes from the hierarchy

        Args:
            icd_code (str): A given code
            topmost_level (bool, defaults to `False`): Flag to get the topmost code in the hierarchy
            allow_billable (bool, defaults to `False): Whether it can return the same code as is (billable)

        Returns:
            (str): Non-billable version of the given code (top level hierarchy)
        """

        # Split on period
        code_split = icd_code.split('.')

        # If we want only the top most level of the code
        # return of he first part of the split.
        # Otherwise, randomly select a non-billable code from the
        # hierarchy
        if len(code_split) == 1 or topmost_level:
            return code_split[0]
        else:
            # Randomly select non-billable code by shortening the length
            category = random.choice(range(0, len(code_split[1]) + int(allow_billable)))
            return code_split[0] + '.' * min(category, 1) + code_split[1][0:category]

    def get_mapping_function(self, billable_probability, top_non_probability, mixed_non_probability) -> Callable:
        """
        We can keep the codes as is, replace with top most level no billable
        codes or have a mixed set. Return the appropriate function based on the
        probabilities
        Args:
            billable_probability (float): The probability of mapping the sequence into only a billable sequence. This
            would mean keeping the sequence as is
            top_non_probability (float): The probability of mapping the sequence into the top most level non-billable
            sequence of codes.
            mixed_non_probability (float): The probability of mapping the sequence into a mix of various non-billable
            and billable sequence of codes.

        Returns:
            (Callable): Returns the function that can be used to map icd codes
        """
        mapping_functions = [
            self.get_billable_code,
            partial(self.get_non_billable_code, topmost_level=True),
            partial(self.get_non_billable_code, allow_billable=True)
        ]
        return np.random.choice(
            mapping_functions,
            p=[billable_probability, top_non_probability, mixed_non_probability]
        )

    def transform_codes(self, icd_code: str, lowercase: bool) -> str:
        """
        Return raw codes or the textual code descriptions

        Args:
            icd_code (str): A given icd code
            lowercase (bool): Whether to lowercase the output

        Returns:
            icd_code (str): String that contains the raw codes or text descriptions
        """

        # If the icd description object is None, return the raw codes
        # else return the description of codes
        if self._icd_descriptions is None:
            return icd_code
        else:
            return (
                self._icd_descriptions.get_description(icd_code).lower()
                if lowercase else self._icd_descriptions.get_description(icd_code)
            )

    def get_converted_code(
            self,
            icd_code: str
    ) -> str:
        """
        Given a set of codes, leave them as is, or convert to relevant codes in their hierarchy.
        Post which the codes can be returned as is or mapped to their textual descriptions.
        Args:
            icd_code (str): An icd code

        Returns:
            icd_code (str): Converted icd code
        """

        # Get the mapping function - to map icd codes to codes in the hierarchy or keep the code as is
        mapping_function = self.get_mapping_function(
            billable_probability=self._billable_probability,
            top_non_probability=self._top_non_probability,
            mixed_non_probability=self._mixed_non_probability
        )

        # Map to textual descriptions
        return self.transform_codes(icd_code=mapping_function(icd_code), lowercase=self._lowercase)

    def get_converted_codes(
            self,
            icd_codes: Sequence[str],
    ) -> List[str]:
        """
        Given a set of codes, leave them as is, or convert to relevant codes in their hierarchy.
        Post which the codes can be returned as is or mapped to their textual descriptions.
        Args:
            icd_codes (Sequence[str]): List of input icd codes

        Returns:
            icd_codes (List[str]): Converted icd codes
        """

        # Get the mapping function - to map icd codes to codes in the hierarchy or keep the code as is
        mapping_function = self.get_mapping_function(
            billable_probability=self._billable_probability,
            top_non_probability=self._top_non_probability,
            mixed_non_probability=self._mixed_non_probability
        )

        # Map codes
        icd_codes = map(mapping_function, icd_codes)

        # Map to textual descriptions
        icd_codes = map(partial(self.transform_codes, lowercase=self._lowercase), icd_codes)

        return list(icd_codes)
