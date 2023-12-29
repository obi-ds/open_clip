import numpy as np
import pandas as pd
from typing import NoReturn

from .multi_tasks import MultiTasks
from .codes import CodeStatusClassificationTask, CodeT2EPredictionTask
from .codes.processing import NegativeCodeCacheSampling

class MultiInstruct(object):

    def __init__(
            self,
            code_status_classification_task: CodeStatusClassificationTask,
            code_t2e_prediction_task: CodeT2EPredictionTask,
            negative_code_cache_sampling: NegativeCodeCacheSampling,
    ):
        self._code_status_classification_task = code_status_classification_task
        self._code_t2e_prediction_task = code_t2e_prediction_task
        self._code_task_object = self.get_code_task_object()
        self._negative_code_cache_sampling = negative_code_cache_sampling

        # TODO: Define task probabilities and task in a better way
        self._tasks = self.get_tasks()
        self._task_probabilities = [0.5, 0.5]


    def get_code_task_object(self):
        if self._code_status_classification_task is not None:
            return self._code_status_classification_task
        elif self._code_t2e_prediction_task is not None:
            return self._code_t2e_prediction_task
        else:
            return None

    def get_tasks(self):
        tasks = list()
        if self._code_status_classification_task is not None:
            tasks.append(MultiTasks.CodeStatusClassificationTask)
        if self._code_t2e_prediction_task is not None:
            tasks.append(MultiTasks.CodeT2EPredictionTask)
        return tasks


    def get_code_status_classification_instructions(
            self,
            encounter_history,
            all_positives,
            positives,
            k_shot,
            sort_position,
            encounter_negatives
    ):
        instruction_samples, instructions = self._code_status_classification_task.get_task_instructions(
            encounter_history=encounter_history,
            all_positives=all_positives,
            positives=positives,
            k_shot=k_shot,
            sort_position=sort_position,
            encounter_negatives=encounter_negatives
        )

        return instruction_samples, instructions

    def get_code_t2e_prediction_instructions(
            self,
            encounter_history,
            all_positives,
            positives,
            k_shot,
            sort_position,
            encounter_negatives
    ):
        instruction_samples, instructions = self._code_t2e_prediction_task.get_task_instructions(
            encounter_history=encounter_history,
            all_positives=all_positives,
            positives=positives,
            k_shot=k_shot,
            sort_position=sort_position,
            encounter_negatives=encounter_negatives
        )

        return instruction_samples, instructions

    def load_data_for_code_task(
            self,
            patient_id,
            current_time,
            past_time_delta,
            future_time_delta,
            use_log_position,
            time_difference_normalize
    ):

        full_encounter_history = self._code_task_object.get_full_encounter_history(
            patient_id=patient_id,
            current_time=current_time,
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )

        return self._code_task_object.load_encounter_history_for_task(
            encounter_history=full_encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta,
        )

    def filter_encounter_history_by_selected_samples(
            self,
            encounter_history: pd.DataFrame,
            selected_samples: pd.DataFrame,
    ) -> NoReturn:
        """

        Args:
            encounter_history (pd.DataFrame): The encounter history of the patient
            selected_samples (pd.DataFrame)): The sampled encounters (either for positives/negatives)

        """
        code_filter = ~(
            encounter_history[self._code_task_object.code_column]
            .isin(selected_samples[self._code_task_object.code_column])
        )
        position_filter = ~(
            encounter_history[self._code_task_object.code_column]
            .isin(selected_samples[self._code_task_object.code_column])
        )
        return encounter_history[code_filter | position_filter]

    def get_filtered_encounter_history_and_negatives(
            self,
            encounter_history,
            encounter_negatives,
            instruction_samples
    ):
        # Remove sampled positives from encounter history - so we don't resample the same thing
        encounter_history = self.filter_encounter_history_by_selected_samples(
            encounter_history=encounter_history,
            selected_samples=instruction_samples[
                instruction_samples[self._code_task_object.instruction_label] == True
                ]
        )
        encounter_negatives = self.filter_encounter_history_by_selected_samples(
            encounter_history=encounter_negatives,
            selected_samples=instruction_samples[
                instruction_samples[self._code_task_object.instruction_label] == False
                ]
        )
        return encounter_history, encounter_negatives

        
    def get_code_task_encounter_negatives_from_random(self, encounter_history):
        position_range = np.arange(
            encounter_history[self._code_task_object.position_column].min(),
            encounter_history[self._code_task_object.position_column].max() + 1,
            step=0.5
        )
        return self._code_task_object.get_random_encounters(
            size=len(encounter_history),
            position_range=position_range,
            exclude_codes=pd.Series([])
        )

    def get_code_task_encounter_negatives(self, encounter_history, patient_id):
        # Sample from cache - if cache is empty - sample randomly
        encounter_negatives = self._negative_code_cache_sampling.get_code_task_encounter_negatives_from_cache_and_update(
            encounter_history=encounter_history,
            patient_id=patient_id
        )
        if not encounter_negatives:
            return self.get_code_task_encounter_negatives_from_random(encounter_history=encounter_history)
        else:
            return encounter_negatives

    def process_sample_from_args(self, sample, args):
        number_of_instructions = 5
        k_shot_choices = [0]
        sort_code_status_position = True
        sort_code_t2e_position = True
        patient_id = sample[args.patient_id_column]

        if self._code_task_object is not None:
            # Get the encounter data
            encounter_history, all_positives, positives = self.load_data_for_code_task(
                patient_id=patient_id,
                current_time=sample[args.sample_result_date_column],
                past_time_delta=args.past_time_delta,
                future_time_delta=args.future_time_delta,
                use_log_position=args.use_log_position,
                time_difference_normalize=args.time_difference_normalize
            )
            # Sample negatives - either from cache or randomly
            encounter_negatives = self.get_code_task_encounter_negatives(
                encounter_history=encounter_history,
                patient_id=patient_id
            )
        else:
            encounter_history, all_positives, positives = None, None, None
            encounter_negatives = None

        # TODO: Adding sorting by position code

        multi_task_instructions = list()
        for _ in range(number_of_instructions):
            instruction_task = np.random.choice(self._tasks, p=self._task_probabilities)
            if instruction_task == MultiTasks.CodeStatusClassificationTask:
                instruction_samples, instructions = self.get_code_status_classification_instructions(
                    encounter_history=encounter_history,
                    all_positives=all_positives,
                    positives=positives,
                    k_shot=np.random.choice(k_shot_choices),
                    sort_position=sort_code_status_position,
                    encounter_negatives=encounter_negatives
                )
                encounter_history, encounter_negatives = self.get_filtered_encounter_history_and_negatives(
                    encounter_history=encounter_history,
                    encounter_negatives=encounter_negatives,
                    instruction_samples=instruction_samples
                )

            elif instruction_task == MultiTasks.CodeT2EPredictionTask:
                instruction_samples, instructions = self.get_code_t2e_prediction_instructions(
                    encounter_history=encounter_history,
                    all_positives=all_positives,
                    positives=positives,
                    k_shot=np.random.choice(k_shot_choices),
                    sort_position=sort_code_t2e_position,
                    encounter_negatives=encounter_negatives
                )
                encounter_history, encounter_negatives = self.get_filtered_encounter_history_and_negatives(
                    encounter_history=encounter_history,
                    encounter_negatives=encounter_negatives,
                    instruction_samples=instruction_samples
                )
            else:
                raise ValueError('Invalid task')
            multi_task_instructions += instructions

        return multi_task_instructions

