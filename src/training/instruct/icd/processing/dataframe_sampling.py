"""Sample dataframe rows based on scores and scoring function"""
from typing import Optional

import numpy as np
import pandas as pd

from .utils import normal_choice


class DataFrameSampling(object):
    """
    Sample from a given dataframe using the scores present in the scores
    column of the dataframe
    """

    @staticmethod
    def get_sample_size(
            dataframe_size: int,
            full_sequence_probability: float,
            single_sequence_probability: float
    ) -> int:
        """
        Calculate the number of samples/icd codes we are going to extract
        for a given patient from their history
        Args:
            dataframe_size (int): The number of codes in the patient history
            full_sequence_probability (float): Probability of sampling all codes
            single_sequence_probability (float): Probability of sampling a single code

        Returns:
            sample_size (int): The number of samples to extract from patient history
        """
        mixed_size = 1 if dataframe_size == 1 else normal_choice(np.arange(1, dataframe_size))
        sample_choices = [dataframe_size, 1, mixed_size]
        sample_size = np.random.choice(
            sample_choices,
            p=[
                full_sequence_probability,
                single_sequence_probability,
                (1 - (full_sequence_probability + single_sequence_probability))
            ]
        )
        return sample_size

    @staticmethod
    def sample_dataframe(
            dataframe: pd.DataFrame,
            sample_size: int,
            weights: Optional[pd.Series]
    ) -> pd.DataFrame:
        """
        Sample dataframe. Use the scores as sampling weights
        Args:
            dataframe (pd.DataFrame): The input dataframe
            sample_size (int): The number of elements to sample
            weights (Optional[pd.Series]): The score for each element

        Returns:
            (pd.DataFrame): Sampled dataframe
        """
        if weights is None:
            return dataframe.sample(n=sample_size)
        else:
            return dataframe.sample(n=sample_size, weights=weights)
