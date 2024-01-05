import numpy as np
import pandas as pd
from collections import Counter

class NegativeCodeCacheSampling(object):

    def __init__(
            self,
            code_task_negative_cache_size: int,
            minimum_encounter_size: int
    ):
        self._code_task_negative_cache = {}
        self._code_task_negative_cache_size = code_task_negative_cache_size
        self._code_task_negative_cache_counts = Counter()
        self._minimum_encounter_size = minimum_encounter_size

    def update_cache_code_task_negatives(self, encounter_history: pd.DataFrame, patient_id: str):
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

    def get_code_task_encounter_negatives_from_cache(self, patient_id):
        cached_patient_ids = self._code_task_negative_cache.keys() - {patient_id}
        # Cache is empty - return empty cache
        if not cached_patient_ids:
            return None
        else:
            # Sample from cache
            sample_cached_patient_id = np.random.choice(list(cached_patient_ids))
            # Keep track of which encounter history was used to sample the negatives from and how many times
            self._code_task_negative_cache_counts[sample_cached_patient_id] += 1
            return self._code_task_negative_cache[sample_cached_patient_id]

    def get_code_task_encounter_negatives_from_cache_and_update(self, encounter_history, patient_id):
        # Get encounter negatives and update cache
        encounter_negatives = self.get_code_task_encounter_negatives_from_cache(patient_id=patient_id)
        if len(encounter_history) >= self._minimum_encounter_size:
            self.update_cache_code_task_negatives(encounter_history=encounter_history, patient_id=patient_id)
        return encounter_negatives