"""Sampling negative codes for a patient"""
import numpy as np
import pandas as pd
from typing import Optional, Sequence


class NegativeCodeCacheSampling(object):
    """
    Sampling from cached encounter histories
    """

    def __init__(
            self,
            code_task_negative_cache_size: int,
            minimum_encounter_size: int,
            minimum_negatives_size: int = 10,
            source_file: Optional[str] = None,
    ):
        self._code_task_negative_cache = {}
        self._last_sampled_cache_id = None
        self._code_task_negative_cache_size = code_task_negative_cache_size
        self._minimum_encounter_size = minimum_encounter_size
        self._minimum_negatives_size = minimum_negatives_size

        if source_file is None:
            source_file = '/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv'
            raw_codes = pd.read_csv(source_file, encoding='ISO-8859-1').phecode
            self._codes = set(raw_codes)
        else:
            raise NotImplementedError('Custom source file yet to be implemented')

    def update_negatives_cache(self, encounter_history: pd.DataFrame, patient_id: str):
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
                del self._code_task_negative_cache[self._last_sampled_cache_id]
                self._code_task_negative_cache[patient_id] = encounter_history

    def get_negatives_from_cache(self, patient_id) -> Optional[pd.DataFrame]:
        """
        Sample an encounter history from the cache - we can then sample codes
        from this and use it as negative samples for the current patient.

        Args:
            patient_id (str): The id pf the patient

        Returns:
            (pd.DataFrame): The encounter history of another patient.
            or None - if cache is empty
        """
        cached_patient_ids = self._code_task_negative_cache.keys() - {patient_id}
        # Cache is empty - return empty cache
        if not cached_patient_ids:
            return None
        else:
            # Sample from cache
            sample_cached_patient_id = np.random.choice(list(cached_patient_ids))
            # Keep track of which encounter history was used to sample the negatives last
            self._last_sampled_cache_id = sample_cached_patient_id
            return self._code_task_negative_cache[sample_cached_patient_id]

    def get_negatives_from_cache_and_process(
            self,
            encounter_negatives: pd.DataFrame,
            exclude_codes: Sequence[str],
            code_column: str = 'phecode',
            position_column: str = 'position'
    ) -> Optional[pd.DataFrame]:
        """
        Sample an encounter history from the cache - we can then sample codes
        from this and use it as negative samples for the current patient.

        Args:
            encounter_negatives (str): The negatives returned from cache
            exclude_codes (Sequence[str]): Remove these codes from the sampled encounter negatives
            code_column (str, defaults to `phecode`): The column that contains codes
            position_column (str, defaults to `position`): The column that contains positions

        Returns:
            (pd.DataFrame): The encounter history of another patient.
            or None - if cache is empty
        """
        # Get encounter negatives and update cache
        if encounter_negatives is None:
            return self.get_random_negatives(
                exclude_codes=set(exclude_codes),
                size=self._minimum_negatives_size,
                code_column=code_column,
                position_column=position_column
            )
        else:
            encounter_negatives = self.exclude_codes_from_dataframe(
                encounter_dataframe=encounter_negatives,
                exclude_codes=exclude_codes,
                code_column=code_column
            )
            if encounter_negatives[code_column].nunique() < self._minimum_negatives_size:
                size = self._minimum_negatives_size - encounter_negatives[code_column].nunique()
                random_negatives = self.get_random_negatives(
                    exclude_codes=set(exclude_codes).union(encounter_negatives[code_column]),
                    size=size,
                    code_column=code_column,
                    position_column=position_column
                )
                encounter_negatives = self.concatenate_negatives(encounter_negatives, random_negatives)
            return encounter_negatives

    def get_encounter_negatives_from_cache_process_and_update(
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
        encounter_negatives = self.get_negatives_from_cache(patient_id=patient_id)

        # Get encounter negatives and update cache
        encounter_negatives = self.get_negatives_from_cache_and_process(
            encounter_negatives=encounter_negatives,
            exclude_codes=exclude_codes,
            code_column=code_column,
            position_column=position_column
        )

        # Update cache
        if encounter_history[code_column].nunique() >= self._minimum_encounter_size:
            self.update_negatives_cache(encounter_history=encounter_history, patient_id=patient_id)

        return encounter_negatives

    def get_random_negatives(
            self,
            exclude_codes: set,
            size: int,
            code_column: str = 'phecode',
            position_column: str = 'position'
    ):
        """
        Sample negatives randomly

        Args:
            size (int): The number of negatives to sample
            exclude_codes (Sequence[str]): Remove these codes from the sampled encounter negatives
            code_column (str, defaults to `phecode`): The column that contains codes
            position_column (str, defaults to `position`): The column that contains positions

        Returns:
            (pd.DataFrame): A dataframe containing randomly sampled negatives
        """
        negatives = self._codes.difference(exclude_codes)
        return pd.DataFrame(
            {code_column: np.random.choice(list(negatives), size, replace=False), position_column: 0}
        )

    @staticmethod
    def exclude_codes_from_dataframe(
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
