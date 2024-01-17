import numpy as np
import pandas as pd
from typing import NoReturn, Tuple, List

import torch

from .multi_tasks import MultiTasks
from .multi_instruct_tokenizer import MultiInstructTokenizer
from .codes import CodeStatusClassificationTask, CodeT2EPredictionTask
from .codes.processing import NegativeCodeCacheSampling

class MultiInstruct(object):

    def __init__(
            self,
            code_status_classification_task_info: Tuple[CodeStatusClassificationTask, float, List[int]],
            code_t2e_prediction_task_info: Tuple[CodeT2EPredictionTask, float, List[int]],
            negative_code_cache_sampling: NegativeCodeCacheSampling,
            multi_instruct_tokenizer: MultiInstructTokenizer,
            task_object_key: str = 'task_object',
            k_shot_choices_key: str = 'k_shot_choices'
    ):
        self._code_status_classification_task_info = code_status_classification_task_info
        self._code_t2e_prediction_task_info = code_t2e_prediction_task_info
        self._negative_code_cache_sampling = negative_code_cache_sampling
        self._multi_instruct_tokenizer = multi_instruct_tokenizer
        self._task_object_key = task_object_key
        self._k_shot_choices_key = k_shot_choices_key
        self._tasks, self._task_probabilities, self._task_info = self.setup_tasks()
        self._code_task_object = self.get_code_task_object()

    def get_code_task_object(self):
        if self._code_status_classification_task_info is not None:
            return self._task_info[MultiTasks.CodeStatusClassificationTask][self._task_object_key]
        elif self._code_t2e_prediction_task_info is not None:
            return self._task_info[MultiTasks.CodeT2EPredictionTask][self._task_object_key]
        else:
            return None

    def setup_tasks(self):
        tasks = list()
        task_probabilities =  list()
        task_info =  dict()
        if self._code_status_classification_task_info is not None:
            tasks.append(MultiTasks.CodeStatusClassificationTask)
            task_probabilities.append(self._code_status_classification_task_info[1])
            task_info[MultiTasks.CodeStatusClassificationTask] =  {
                self._k_shot_choices_key: self._code_status_classification_task_info[2],
                self._task_object_key: self._code_status_classification_task_info[0]
            }
        if self._code_t2e_prediction_task_info is not None:
            tasks.append(MultiTasks.CodeT2EPredictionTask)
            task_probabilities.append(self._code_t2e_prediction_task_info[1])
            task_info[MultiTasks.CodeT2EPredictionTask] = {
                self._k_shot_choices_key: self._code_t2e_prediction_task_info[2],
                self._task_object_key: self._code_t2e_prediction_task_info[0]
            }
        return tasks, task_probabilities, task_info

    def get_code_status_classification_instructions(
            self,
            encounter_history,
            all_positives,
            positives,
            k_shot,
            sort_position,
            encounter_negatives
    ):
        instruction_samples, instructions = self._task_info[
            MultiTasks.CodeStatusClassificationTask
        ][self._task_object_key].get_task_instructions(
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
        instruction_samples, instructions = self._task_info[
            MultiTasks.CodeT2EPredictionTask
        ][self._task_object_key].get_task_instructions(
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
        # If encounter_negatives becomes empty - the code will run
        # but will randomly sample negatives
        encounter_negatives = self.filter_encounter_history_by_selected_samples(
            encounter_history=encounter_negatives,
            selected_samples=instruction_samples[
                instruction_samples[self._code_task_object.instruction_label] == False
                ]
        )
        return encounter_history, encounter_negatives

    def get_code_task_encounter_negatives_from_random(self, encounter_history, encounter_length=20):
        if encounter_history.empty:
            position_range = np.arange(0, 1, step=0.5)
        else:
            position_range = np.arange(
                encounter_history[self._code_task_object.position_column].min(),
                encounter_history[self._code_task_object.position_column].max() + 1,
                step=0.5
            )
        return self._code_task_object.get_random_encounters(
            size=encounter_length,
            position_range=position_range,
            exclude_codes=pd.Series([])
        )

    def get_code_task_encounter_negatives(self, encounter_history, patient_id):
        # Sample from cache - if cache is empty - sample randomly
        encounter_negatives = self._negative_code_cache_sampling.get_code_task_encounter_negatives_from_cache_and_update(
            encounter_history=encounter_history,
            patient_id=patient_id
        )
        if encounter_negatives is None:
            return self.get_code_task_encounter_negatives_from_random(
                encounter_history=encounter_history
            )
        else:
            return encounter_negatives

    def get_multi_task_instructions(self, sample, args):
        number_of_instructions = np.random.choice(args.number_of_instructions)
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
        processed_instructions = 0
        while processed_instructions < number_of_instructions:
            instruction_task = np.random.choice(self._tasks, p=self._task_probabilities)
            if instruction_task == MultiTasks.CodeStatusClassificationTask:
                k_shot = min(
                    np.random.choice(
                        self._task_info[MultiTasks.CodeStatusClassificationTask][self._k_shot_choices_key]
                    ),
                    number_of_instructions - processed_instructions - 1
                )
                instruction_samples, instructions = self.get_code_status_classification_instructions(
                    encounter_history=encounter_history,
                    all_positives=all_positives,
                    positives=positives,
                    k_shot=k_shot,
                    sort_position=sort_code_status_position,
                    encounter_negatives=encounter_negatives
                )
                encounter_history, encounter_negatives = self.get_filtered_encounter_history_and_negatives(
                    encounter_history=encounter_history,
                    encounter_negatives=encounter_negatives,
                    instruction_samples=instruction_samples
                )

            elif instruction_task == MultiTasks.CodeT2EPredictionTask:
                k_shot = min(
                    np.random.choice(
                        self._task_info[MultiTasks.CodeT2EPredictionTask][self._k_shot_choices_key]
                    ),
                    number_of_instructions - processed_instructions - 1
                )
                instruction_samples, instructions = self.get_code_t2e_prediction_instructions(
                    encounter_history=encounter_history,
                    all_positives=all_positives,
                    positives=positives,
                    k_shot=k_shot,
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
            processed_instructions += k_shot + 1
            multi_task_instructions += instructions
        return multi_task_instructions

    def process_sample_from_args(self, sample, args):
        multi_task_instructions = self.get_multi_task_instructions(sample=sample, args=args)
        if self._multi_instruct_tokenizer is None:
            return multi_task_instructions

        input_ids, labels = self._multi_instruct_tokenizer.get_tokens(
            multi_task_instructions=multi_task_instructions, return_tensor=True
        )
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)

        return torch.cat([input_ids, labels])

