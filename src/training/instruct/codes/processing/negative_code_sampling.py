"""Sampling negative codes for a patient"""
import numpy as np
import pandas as pd
from collections import Counter
import os.path as path
from typing import Optional, Sequence


class NegativeCodeCacheSampling(object):
    """
    Sampling from cached encounter histories
    """

    def __init__(
            self,
            code_task_negative_cache_size: int,
            minimum_encounter_size: int,
            minimum_negatives_size: int = 50,
            source_file: Optional[str] = None,
    ):
        self._code_task_negative_cache = {}
        self._code_task_negative_cache_size = code_task_negative_cache_size
        self._code_task_negative_cache_counts = Counter()
        self._minimum_encounter_size = minimum_encounter_size
        self._minimum_negatives_size = minimum_negatives_size

        if source_file is None:
            source_file = path.dirname(path.abspath(__file__)) + '/../descriptions/phe_codes_flat.csv'
            raw_codes = pd.read_csv(source_file).phecode
            # TODO - Add code convert
            # self._codes = set(self._code_convert.get_converted_codes(raw_codes))
            self._codes = set(raw_codes)
        else:
            raise NotImplementedError('Custom source file yet to be implemented')

    def update_cache_code_task_negatives(self, encounter_history: pd.DataFrame, patient_id: str):
        """
        Update cache with this encounter history

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            patient_id (str): The ID of the patient
        """
        # If patient id is already in cache - replace the encounter history
        if patient_id in self._code_task_negative_cache.keys():
            self._code_task_negative_cache[patient_id] = encounter_history
        else:
            # If patient id is not in cache and there is space to add more,
            # add a new entry to the cache.
            if len(self._code_task_negative_cache) < self._code_task_negative_cache_size:
                self._code_task_negative_cache[patient_id] = encounter_history
            # If cache is full - remove based on counts
            else:
                # Remove the encounter history that has been used to sample negatives the most
                remove_patient_id = self._code_task_negative_cache_counts.most_common()[0][0]
                del self._code_task_negative_cache[remove_patient_id]
                del self._code_task_negative_cache_counts[remove_patient_id]
                self._code_task_negative_cache[patient_id] = encounter_history

    def get_code_task_encounter_negatives_from_cache(self, patient_id) -> pd.DataFrame:
        """
        Sample an encounter history from the cache - we can then sample codes
        from this and use it as negative samples for the current patient.

        Args:
            patient_id (str): The id pf the patient

        Returns:
            (pd.DataFrame): The encounter history of another patient.
            or an empty dataframe - if cache is empty
        """
        cached_patient_ids = self._code_task_negative_cache.keys() - {patient_id}
        # Cache is empty - return empty cache
        if not cached_patient_ids:
            return pd.DataFrame()
        else:
            # Sample from cache
            sample_cached_patient_id = np.random.choice(list(cached_patient_ids))
            # Keep track of which encounter history was used to sample the negatives from and how many times
            self._code_task_negative_cache_counts[sample_cached_patient_id] += 1
            return self._code_task_negative_cache[sample_cached_patient_id]

    def get_code_task_encounter_negatives_from_cache_and_update(
            self,
            encounter_history: pd.DataFrame,
            patient_id: str,
            exclude_codes: Sequence[str],
            code_column: str = 'phecode',
            position_column: str = 'position'
    ):
        """
        Sample an encounter history from the cache - we can then sample codes
        from this and use it as negative samples for the current patient.
        Once this is done - we update the cache

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            patient_id (str): The id of the patient
            exclude_codes (Sequence[str]): Remove these codes from the sampled encounter negatives
            code_column (str, defaults to `phecode`): The column that contains codes
            position_column (str, defaults to `position`): The column that contains positions

        Returns:

        """
        # Get encounter negatives and update cache
        encounter_negatives = self.get_code_task_encounter_negatives_from_cache(patient_id=patient_id)
        if encounter_negatives.empty:
            encounter_negatives = pd.DataFrame(
                {code_column: [], position_column: []}
            )

        # Exclude any codes from the negatives dataframe that is retrieved from cache
        encounter_negatives = self.get_exclude_codes_dataframe(
            encounter_dataframe=encounter_negatives,
            exclude_codes=exclude_codes,
            code_column=code_column
        )

        # Add random negative codes if it doesn't meet size requirements
        if len(encounter_negatives) < self._minimum_negatives_size:
            negatives = (
                self._codes
                .difference(set(exclude_codes))
                .difference(set(encounter_negatives[code_column]))
            )
            size = self._minimum_negatives_size - len(encounter_negatives)
            random_negatives = pd.DataFrame(
                {code_column: np.random.choice(list(negatives), size, replace=False), position_column: 0}
            )
            encounter_negatives = self.concatenate_negatives(encounter_negatives, random_negatives)

        # Update cache
        if len(encounter_history) >= self._minimum_encounter_size:
            self.update_cache_code_task_negatives(encounter_history=encounter_history, patient_id=patient_id)

        return encounter_negatives

    @staticmethod
    def get_exclude_codes_dataframe(
            encounter_dataframe: pd.DataFrame,
            exclude_codes: Sequence[str],
            code_column: str,
    ) -> pd.DataFrame:
        """
        Given an input dataframe - return the original dataframe without the
        codes given in exclude codes - basically exclude these codes from the
        dataframe and return
        Args:
            encounter_dataframe (pd.DataFrame): The input dataframe
            exclude_codes (Sequence[str]): The codes to exclude
            code_column (str): The column that contains the diagnostic codes

        Returns:
            (pd.DataFrame): Dataframe with code excluded
        """
        return encounter_dataframe[~encounter_dataframe[code_column].isin(exclude_codes)]

    @staticmethod
    def concatenate_negatives(
            encounter_negatives: pd.DataFrame,
            random_negatives: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Concatenate the positive and negative samples
        Args:
            encounter_negatives (pd.DataFrame): The sampled positives
            random_negatives (pd.DataFrame): The sampled negatives

        Returns:
            (pd.DataFrame): The concatenated dataframe
        """
        if not encounter_negatives.empty and not random_negatives.empty:
            return pd.concat([encounter_negatives, random_negatives])
        elif not random_negatives.empty:
            return random_negatives
        elif not encounter_negatives.empty:
            return encounter_negatives
        else:
            raise ValueError('Both dataframes are empty')
