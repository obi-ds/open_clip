"""Sampling negative codes for a patient"""
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union

from .encounter_dataframe_process import EncounterDataframeProcess


class NegativeCodeCacheSampling(object):
    """
    Sampling from cached encounter histories
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            code_task_negative_cache_size: int,
            minimum_encounter_size: int,
            minimum_negatives_size: int = 10,
            source_file: Optional[str] = None,
    ):
        self._encounter_dataframe_process = encounter_dataframe_process
        self._code_task_negative_cache = set()
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

    def update_cache(self, patient_id: str):
        """
        Update cache with this patient id

        Args:
            patient_id (str): The ID of the patient
        """
        if len(self._code_task_negative_cache) >= self._code_task_negative_cache_size:
            self._code_task_negative_cache.remove(self._last_sampled_cache_id)
        self._code_task_negative_cache.add(patient_id)

    def get_patient_from_cache(self, patient_id) -> Optional[str]:
        """
        Sample an encounter history from the cache - we can then sample codes
        from this and use it as negative samples for the current patient.

        Args:
            patient_id (str): The id pf the patient

        Returns:
            (pd.DataFrame): The id of the cached patient
        """
        cached_patient_ids = self._code_task_negative_cache - {patient_id}
        # Cache is empty - return empty cache
        if not cached_patient_ids:
            return None
        else:
            # Sample from cache
            sample_cached_patient_id = np.random.choice(list(cached_patient_ids))
            # Keep track of which encounter history was used to sample the negatives last
            self._last_sampled_cache_id = sample_cached_patient_id
            return sample_cached_patient_id

    def get_encounter_negatives_from_cache_process_and_update(
            self,
            patient_id: str,
            current_time: str,
            update_cache: bool,
            use_log_position: bool,
            time_difference_normalize: int,
            exclude_codes: Sequence[str],
            code_column: str = 'phecode',
            position_column: str = 'position'
    ):
        """
        Sample an encounter history from the cache - we can then sample codes
        from this and use it as negative samples for the current patient.
        Once this is done - we update the cache

        Args:
            patient_id (str): The id of the patient
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            update_cache (bool): Whether to update the cache with this id
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)
            exclude_codes (Sequence[str]): Remove these codes from the sampled encounter negatives
            code_column (str, defaults to `phecode`): The column that contains codes
            position_column (str, defaults to `position`): The column that contains positions

        Returns:

        """
        negative_patient_id = self.get_patient_from_cache(patient_id=patient_id)

        if (
                negative_patient_id is None
                or not self._encounter_dataframe_process.check_patient_id(patient_id=negative_patient_id)
        ):
            encounter_negatives = self.get_random_negatives(
                exclude_codes=set(exclude_codes),
                size=self._minimum_negatives_size,
                code_column=code_column,
                position_column=position_column
            )
        else:
            encounter_negatives = self.get_encounter_history(
                patient_id=negative_patient_id,
                current_time=current_time,
                use_log_position=use_log_position,
                time_difference_normalize=time_difference_normalize,
                code_column=code_column
            )
            encounter_negatives = self.process_negatives(
                encounter_negatives=encounter_negatives,
                exclude_codes=exclude_codes,
                code_column=code_column,
                position_column=position_column
            )

        if update_cache:
            # Update cache
            self.update_cache(patient_id=patient_id)

        return encounter_negatives

    def process_negatives(
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
            encounter_negatives (str): The negative encounter history
            exclude_codes (Sequence[str]): Remove these codes from the sampled encounter negatives
            code_column (str, defaults to `phecode`): The column that contains codes
            position_column (str, defaults to `position`): The column that contains positions

        Returns:
            (pd.DataFrame): The encounter history of another patient.
            or None - if cache is empty
        """
        negative_positions = encounter_negatives[position_column]
        # Get encounter negatives and update cache
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
                positions=negative_positions,
                code_column=code_column,
                position_column=position_column
            )
            encounter_negatives = self.concatenate_negatives(encounter_negatives, random_negatives)
        return encounter_negatives

    def get_encounter_history(
            self,
            patient_id: Union[str, int],
            current_time: str,
            use_log_position: bool,
            time_difference_normalize: int,
            code_column: str = 'phecode',
    ) -> pd.DataFrame:
        """
        Return the encounter history of a patient - if the patient does not have an encounter history
        return a random history (temporary fix)

        Args:
            patient_id (Union[str, int]): The unique id of the patient we need to process
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)
            code_column (str, defaults to `phecode`): The column that contains codes

        Returns:
            (pd.DataFrame): The encounter history
        """
        encounter_history = self._encounter_dataframe_process.get_patient_encounter_history_with_position(
            patient_id=patient_id,
            current_time=current_time,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )
        encounter_history.dropna(subset=code_column, inplace=True)
        return encounter_history

    def get_random_negatives(
            self,
            exclude_codes: set,
            size: int,
            positions: Optional = None,
            code_column: str = 'phecode',
            position_column: str = 'position'
    ):
        """
        Sample negatives randomly

        Args:
            size (int): The number of negatives to sample
            exclude_codes (Sequence[str]): Remove these codes from the sampled encounter negatives
            positions (Optional): Use pre-defined position values
            code_column (str, defaults to `phecode`): The column that contains codes
            position_column (str, defaults to `position`): The column that contains positions

        Returns:
            (pd.DataFrame): A dataframe containing randomly sampled negatives
        """
        negatives = self._codes.difference(exclude_codes)

        if positions is None:
            position_column_value = 0
        else:
            position_column_value = np.random.choice(positions, size=size)
        return pd.DataFrame(
            {
                code_column: np.random.choice(list(negatives), size, replace=False),
                position_column: position_column_value
            }
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
