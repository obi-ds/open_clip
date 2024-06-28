"""Sampling negative codes for a patient"""
import pygtrie
import numpy as np
import pandas as pd
from treelib import Node, Tree
from typing import Optional, Union, Tuple

from .encounter_dataframe_process import EncounterDataframeProcess


class NegativeCodeCacheSampling(object):
    """
    Sampling from cached encounter histories
    """

    def __init__(
            self,
            encounter_dataframe_process: EncounterDataframeProcess,
            negatives_type,
            code_task_negative_cache_size: int,
            source_file: Optional[str] = None,
            code_column: str = 'phecode',
            idf_column: str = 'idf',
            position_column: str = 'position'
    ):
        self._encounter_dataframe_process = encounter_dataframe_process
        self._negatives_type = negatives_type
        if self._negatives_type == 'random_cached':
            self._random_cached_weight = 1.0
        self._code_task_negative_cache = set()
        self._last_sampled_cache_id = None
        self._code_task_negative_cache_size = code_task_negative_cache_size
        self._code_column = code_column
        self._position_column = position_column
        self._idf_column = idf_column

        if source_file is None:
            source_file = '/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv'
            idf_file = '/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/cardiac/idfs.csv'
            # Used for random negatives - get codes that have idf value
            idf = pd.read_csv(idf_file)
            codes = pd.read_csv(source_file, encoding='ISO-8859-1')
            self._codes = pd.merge(codes, idf, on='phecode', how='left')
            self._codes[self._idf_column].fillna(idf[self._idf_column].mean(), inplace=True)

        else:
            raise NotImplementedError('Custom source file yet to be implemented')

        self._trie = self.build_trie(codes=self._codes[self._code_column])
        self._tree = self.build_tree(codes=self._codes[self._code_column])

    def get_encounter_negatives_from_cache_process_and_update(
            self,
            patient_id: str,
            current_time: str,
            prediction_range: Tuple[int, int],
            use_log_position: bool,
            time_difference_normalize: int,
            minimum_negatives_size: int,
            exclude_codes: pd.Series,
            weights: Optional = None
    ):
        """
        Sample an encounter history from the cache - we can then sample codes
        from this and use it as negative samples for the current patient.
        Once this is done - we update the cache

        Args:
            patient_id (str): The id of the patient
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            prediction_range (Tuple[int, int]): The range we are making prediction in
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)
            exclude_codes (pd.Series): Remove these codes from the sampled encounter negatives
            minimum_negatives_size (int): How many negatives to sample
            weights (): Weights for sampling

        Returns:

        """
        # Retrieve one of the cached patients
        negative_patient_id = self.get_patient_from_cache(patient_id=patient_id)

        # If cache is empty, or if the cached id does not have any encounter data or  if
        # we only want to sample random negatives
        if (
                negative_patient_id is None
                or not self._encounter_dataframe_process.check_patient_id(patient_id=negative_patient_id)
                or self._negatives_type == 'random'
        ):
            encounter_negatives = self.get_random_negatives(
                exclude_codes=exclude_codes,
                size=minimum_negatives_size,
                prediction_range=prediction_range,
                weights=weights
            )
        # Get negatives from the encounter history of the cached patient id
        else:
            encounter_negatives = self.get_encounter_history(
                patient_id=negative_patient_id,
                current_time=current_time,
                use_log_position=use_log_position,
                time_difference_normalize=time_difference_normalize,
            )
            # Once we have the encounter negatives - we compute position values and
            # also exclude codes from the encounter history. If the negatives is to
            # small we add some random negatives to it. We try and return a set
            # that consists of random and cached negatives of equal numbers.
            encounter_negatives = self.process_negatives(
                encounter_negatives=encounter_negatives,
                minimum_negatives_size=minimum_negatives_size,
                exclude_codes=exclude_codes,
                prediction_range=prediction_range,
                weights=weights
            )

        # Add the current patient id to cache
        if self._encounter_dataframe_process.check_patient_id(patient_id=patient_id):
            self.update_cache(patient_id=patient_id)

        return encounter_negatives

    def build_trie(self, codes):
        """
        Build the trie datastructure

        Args:
            codes:

        Returns:

        """
        trie = pygtrie.StringTrie()
        # Add root codes to trie (e.g. CV, DE etc)
        for code in codes.str.split('_').str[0].unique():
            trie[code] = code
        # Add all other codes
        for code in codes:
            # Representation essentially contains the path to get to the code
            key = '/'.join(self.get_code_representation(code))
            trie[key] = code
        return trie

    def build_tree(self, codes):
        """
        Build the tree datastructure - useful when we want parent child relations

        Args:
            codes:

        Returns:

        """
        tree = Tree()
        # Create the root node - the next level will have
        # different disease categories - e.g. cardiology, neurological etc.
        tree.create_node("Codes", "root")
        # Add root codes to trie (e.g. CV, DE etc)
        for code in codes.str.split('_').str[0].unique():
            tree.create_node(code, code, parent='root')
        # Now add all the children
        for code in codes:
            # Representation essentially contains the path to get to the code
            keys = self.get_code_representation(code)
            # For every code - get all it's ancestors - parents, grandparents ...
            # We check if the ancestors are added before adding the code
            for index, key in enumerate(keys[1:]):
                if tree.contains(key):
                    continue
                # The parent is index - 1 - the code_representation object
                # returns ancestors in order
                tree.create_node(key, key, parent=keys[index])
        return tree

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

    def get_random_negatives(
            self,
            exclude_codes: pd.Series,
            size: int,
            prediction_range: Tuple[int, int],
            weights,
    ):
        """
        Sample negatives randomly. Also exclude any codes that we don't want as negatives

        Args:
            size (int): The number of negatives to sample
            exclude_codes (pd.Series): Remove these codes from the sampled encounter negatives
            prediction_range (Tuple[int, int]): The range we are making prediction in
            weights (): Weights for sampling

        Returns:
            (pd.DataFrame): A dataframe containing randomly sampled negatives
        """
        # From the list of possible codes - exclude ancestors and children of the given code
        negatives = self.exclude_codes_from_dataframe(
            encounter_dataframe=self._codes,
            exclude_codes=exclude_codes,
        )
        if weights is not None:
            weights = weights[negatives[self._code_column]].values

        negatives = negatives.sample(n=size, weights=weights)

        # Add positions to this dataframe
        positions = np.random.choice(
            np.arange(prediction_range[0], prediction_range[1] + 1), size, replace=True
        )
        negatives[self._position_column] = positions

        return negatives

    def process_negatives(
            self,
            encounter_negatives: pd.DataFrame,
            minimum_negatives_size: int,
            exclude_codes: pd.Series,
            prediction_range: Tuple[int, int],
            weights,
    ) -> Optional[pd.DataFrame]:
        """
        Sample an encounter history from the cache - we can then sample codes
        from this and use it as negative samples for the current patient.

        Args:
            encounter_negatives (str): The negative encounter history
            minimum_negatives_size (int): How many negatives to get
            exclude_codes (pd.Series): Remove these codes from the sampled encounter negatives
            prediction_range (Tuple[int, int]): The range we are making prediction in
            weights (): Weights for sampling

        Returns:
            (pd.DataFrame): The encounter history of another patient.
            or None - if cache is empty
        """
        # From the list of codes in the negative patient - exclude ancestors
        # and children of the given code
        encounter_negatives = self.exclude_codes_from_dataframe(
            encounter_dataframe=encounter_negatives,
            exclude_codes=exclude_codes,
        )
        number_unique = self.get_number_of_codes_in_range(
            encounter_history=encounter_negatives,
            prediction_range=prediction_range,
        )

        # We are essentially calculating the number of random negatives we want
        if self._negatives_type == 'random_cached':
            # Set size such that we have equal number of encounter negatives and random
            # negatives
            size = max(int(number_unique * self._random_cached_weight), minimum_negatives_size - number_unique)
        elif self._negatives_type == 'cached':
            # Use only cached negatives - but if the number is too small
            # then add random negatives
            size = max(0, minimum_negatives_size - number_unique)
        else:
            raise ValueError('Invalid negatives type')
        if size == 0:
            return encounter_negatives
        # Get random negatives - exclude codes and the ones already present in the negatives
        random_negatives = self.get_random_negatives(
            exclude_codes=pd.concat([exclude_codes, encounter_negatives[self._code_column]]),
            size=size,
            prediction_range=prediction_range,
            weights=weights
        )
        encounter_negatives = self.concatenate_negatives(encounter_negatives, random_negatives)
        return encounter_negatives

    def get_number_of_codes_in_range(
            self,
            encounter_history: pd.DataFrame,
            prediction_range: Tuple[int, int],
    ) -> pd.DataFrame:
        """
        Return encounters within the current time range

        Args:
            encounter_history:
            prediction_range:

        Returns:

        """

        current_time_period = encounter_history[
            (encounter_history[self._position_column] >= prediction_range[0]) &
            (encounter_history[self._position_column] <= prediction_range[1])
            ]
        return current_time_period[self._code_column].nunique()

    def update_cache(self, patient_id: str):
        """
        Update cache with this patient id

        Args:
            patient_id (str): The ID of the patient
        """
        if len(self._code_task_negative_cache) >= self._code_task_negative_cache_size:
            self._code_task_negative_cache.remove(self._last_sampled_cache_id)
        self._code_task_negative_cache.add(patient_id)

    @staticmethod
    def get_code_representation(code):
        """
        Given a code - get it's trie representation - this would consist of it's parents
        all combined in string form

        Args:
            code:

        Returns:

        """
        if '_' not in code:
            return [code]
        root_split = code.split('_')
        parent_split = code.split('.')
        ancestors = [root_split[0], parent_split[0]]
        if len(parent_split) == 1:
            return ancestors
        sub_code_string = ''
        for char in parent_split[1]:
            sub_code_string += char
            ancestors.append(parent_split[0] + '.' + sub_code_string)
        return ancestors

    def exclude_codes_from_dataframe(
            self,
            encounter_dataframe: Union[pd.DataFrame, pd.Series],
            exclude_codes: pd.Series,
    ) -> pd.DataFrame:
        """
        Given an input dataframe - return the original dataframe without the
        codes given in exclude codes - basically exclude these codes from the
        dataframe and return. We exclude such that the ancestors of the given code
        are excluded, but not it's siblings. We also exclude the children of the
        given code

        Args:
            encounter_dataframe (pd.DataFrame): The input dataframe
            exclude_codes ( pd.Series): The codes to exclude

        Returns:
            (pd.DataFrame): Dataframe with code excluded
        """
        exclude = list()
        for code in exclude_codes:
            # This will return the ancestors for a given code
            key = '/'.join(self.get_code_representation(code))
            # We then exclude all the ancestors
            exclude.extend(self.get_exclude_codes_from_trie(key))
        # Remove rows that contains this expanded set of codes
        if isinstance(encounter_dataframe, pd.DataFrame):
            return encounter_dataframe[~encounter_dataframe[self._code_column].isin(exclude)]
        else:
            return encounter_dataframe[~encounter_dataframe.isin(exclude)]

    def get_exclude_codes_from_trie(self, code):
        """
        Get the codes to excludes based on the trie.
        Exclude parents and children but not siblings

        Args:
            code:

        Returns:

        """
        # The first part excludes the children, the second part excludes the ancestors
        # Prefixes essentially contains the path to get to the code
        return list(self._trie[code:]) + [code_obj[1] for code_obj in self._trie.prefixes(code)]

    def get_encounter_history(
            self,
            patient_id: Union[str, int],
            current_time: str,
            use_log_position: bool,
            time_difference_normalize: int,
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

        Returns:
            (pd.DataFrame): The encounter history
        """
        encounter_history = self._encounter_dataframe_process.get_patient_encounter_history_with_position(
            patient_id=patient_id,
            current_time=current_time,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )
        encounter_history.dropna(subset=self._code_column, inplace=True)
        return encounter_history

    def get_trie(self):
        """
        Return trie datastructure

        Returns:

        """
        return self._trie

    def get_tree(self):
        """
        Return tree datastructure

        Returns:

        """
        return self._tree

    def get_codes(self):
        """
        Return the set of all codes

        Returns:

        """
        return self._codes

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
