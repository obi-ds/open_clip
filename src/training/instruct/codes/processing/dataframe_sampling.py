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
            group_by_column: str = None,
    ) -> pd.DataFrame:
        """
        Sample dataframe. Use the group by function and sample
        all or a collection of groups

        Args:
            dataframe (pd.DataFrame): The input dataframe
            group_by_column (str): The column used to group elements

        Returns:
            (pd.DataFrame): Sampled dataframe
        """

        # Perform groupby operation
        group_by_object = dataframe.groupby(group_by_column)

        # Get the number of groups we are going to sample
        number_of_groups_to_sample = self.get_number_of_groups_to_sample(
            total_number_of_groups=group_by_object.ngroups
        )

        # Based on the number of groups - sample only those groups
        # Return the input dataframe containing only those groups
        return self.sample_dataframe_groups(
            dataframe=dataframe,
            group_by_object=group_by_object,
            number_of_groups_to_sample=number_of_groups_to_sample
        )

    @staticmethod
    def get_number_of_groups_to_sample(total_number_of_groups: int) -> int:
        """
        Given the total number of groups - return the number of groups
        to sample

        Args:
            total_number_of_groups (int): The total number of groups in dataframe

        Returns:
            (int): The number of groups to sample
        """
        # sampling_percentages = np.arange(0.1, 1, 0.1)
        # percentage_sample = int(max(1, np.round(np.random.choice(sampling_percentages * number_of_groups))))
        return np.random.choice([1, total_number_of_groups], p=[0.0, 1.0])

    @staticmethod
    def sample_dataframe_groups(
            dataframe: pd.DataFrame,
            group_by_object,
            number_of_groups_to_sample: int,
    ) -> pd.DataFrame:
        """
        Sample dataframe. Given the number of groups to
        sample - return the groups that contain only those
        many number of groups

        Args:
            dataframe (pd.DataFrame): The input dataframe
            group_by_object (): The pandas group by object
            number_of_groups_to_sample (int): The number of groups to sample

        Returns:
            (pd.DataFrame): Sampled dataframe
        """
        # Get the total number of groups
        total_number_of_groups = group_by_object.ngroups
        if number_of_groups_to_sample == total_number_of_groups:
            return dataframe
        else:
            # Create an index array thing - we will select the number from this
            groups_array = np.arange(total_number_of_groups)
            # Shuffle it - so that we can sample any group
            np.random.shuffle(groups_array)
            # Select only those rows that belong to the groups we specify using the array
            return dataframe[group_by_object.ngroup().isin(groups_array[:number_of_groups_to_sample])]


