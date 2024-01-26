"""
Sample dataframe rows based on scores and scoring function.
This class can be used to sample positives. We can define other
classes that do more complex sampling
"""
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
        Calculate the number of samples/codes we are going to extract
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

    def sample_dataframe(
            self,
            dataframe: pd.DataFrame,
            sample_size: int,
    ) -> pd.DataFrame:
        """
        Sample dataframe. Use the scores as sampling weights

        Args:
            dataframe (pd.DataFrame): The input dataframe
            sample_size (int): The number of elements to sample

        Returns:
            (pd.DataFrame): Sampled dataframe
        """
        weights = self.get_sampling_weights(dataframe=dataframe)
        return dataframe.sample(n=sample_size, weights=weights)

    @staticmethod
    def get_sampling_weights(dataframe) -> np.array:
        # TODO: Eventually we can replace it with smarter sampling
        """
        Randomly sample - no sampling strategy

        Args:
            dataframe (pd.DataFrame): The encounter dataframe

        Returns:
            (np.array): Does not return anything
        """
        return np.ones(len(dataframe))


class GroupBySampling(DataFrameSampling):
    """
    Sample from the groups after doing group by in a given dataframe
    """

    def sample_dataframe(
            self,
            dataframe: pd.DataFrame,
            sample_size: float,
            group_by_column: str = None,
            values_column: str = None
    ) -> pd.DataFrame:
        """
        Sample dataframe. Use the group by function and sample from within the groups

        Args:
            dataframe (pd.DataFrame): The input dataframe
            sample_size (float): The fraction of elements to sample within each group
            group_by_column (str): The column used to group elements
            values_column (str): Calculate the counts within each group for elements belonging to this column

        Returns:
            (pd.DataFrame): Sampled dataframe
        """
        grouped_value_counts = self.get_sampling_weights(
            dataframe=dataframe,
            group_by_column=group_by_column,
            values_column=values_column
        )
        sampled_codes = (
            grouped_value_counts
            .groupby(group_by_column)
            .sample(frac=sample_size, weights=grouped_value_counts)
        )
        return pd.merge(
            sampled_codes,
            dataframe.drop_duplicates(subset=[values_column, group_by_column]),
            how='left',
            on=[group_by_column, values_column]
        )

    @staticmethod
    def get_sampling_weights(
            dataframe: pd.DataFrame,
            group_by_column: str = None,
            values_column: str = None
    ) -> pd.DataFrame:
        """
        Given a dataframe, a group by column and a values column
        Compute the counts of the values in the groups

        Args:
            dataframe (pd.DataFrame): The input dataframe
            group_by_column (str): The column used to group elements
            values_column (str): Calculate the counts within each group for elements belonging to this column

        Returns:
            (pd.DataFrame): Dataframe that contains the counts of the values in each group
        """
        return dataframe.groupby(group_by_column)[values_column].value_counts()
