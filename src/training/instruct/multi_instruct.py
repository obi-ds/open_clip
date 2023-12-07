import numpy as np

from .multi_tasks import MultiTasks
from .codes import CodeStatusClassificationTask, CodeT2EPredictionTask

class MultiInstruct(object):

    def __init__(
            self,
            code_status_classification_task: CodeStatusClassificationTask,
            code_t2e_prediction_task: CodeT2EPredictionTask,
    ):
        self._code_status_classification_task = code_status_classification_task
        self._code_t2e_prediction_task = code_t2e_prediction_task
        self._icd_task_object = self.get_icd_task_object()
        # TODO: Define task probabilities
        self._tasks = self.get_tasks()
        self._task_probabilities = [0.5, 0.5]

    def get_icd_task_object(self):
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
    ):
        return self._code_status_classification_task.process_sample(
            encounter_history=encounter_history,
            all_positives=all_positives,
            positives=positives,
            k_shot=k_shot,
            sort_position=sort_position,
        )

    def get_code_t2e_prediction_instructions(
            self,
            encounter_history,
            all_positives,
            positives,
            k_shot,
            sort_position,
    ):
        return self._code_t2e_prediction_task.process_sample(
            encounter_history=encounter_history,
            all_positives=all_positives,
            positives=positives,
            k_shot=k_shot,
            sort_position=sort_position,
        )

    def load_data_for_icd_task(self, sample, args):

        full_encounter_history = self._icd_task_object.get_full_encounter_history(
            patient_id=sample[args.patient_id_column],
            current_time=sample[args.sample_result_date_column],
            use_log_position=args.use_log_position,
            time_difference_normalize=args.time_difference_normalize
        )

        return self._icd_task_object.load_encounter_history_for_task(
            encounter_history=full_encounter_history,
            past_time_delta=args.past_time_delta,
            future_time_delta=args.future_time_delta,
        )

    def filter_data_for_icd_task(self, encounter_history, position):

        self._code_status_classification_task.filter_encounter_history_by_code_position(
            encounter_history=encounter_history,
            position=position
        )

    def process_sample_from_args(self, sample, args):
        """

        Args:
            sample:
            args:

        Returns:

        """
        number_of_instructions = 5
        k_shot_choices = [0, 1, 2]

        if self._icd_task_object is not None:
            encounter_history, all_positives, positives = self.load_data_for_icd_task(sample=sample, args=args)
        else:
            encounter_history, all_positives, positives = None, None, None

        multi_task_instructions = list()
        for _ in range(number_of_instructions):
            instruction_task = np.random.choice(self._tasks, p=self._task_probabilities)
            if instruction_task == MultiTasks.CodeStatusClassificationTask:
                instructions = self.get_code_status_classification_instructions(
                    encounter_history=encounter_history,
                    all_positives=all_positives,
                    positives=positives,
                    k_shot=np.random.choice(k_shot_choices),
                    sort_position=True
                )

            elif instruction_task == MultiTasks.CodeT2EPredictionTask:
                instructions = self.get_code_t2e_prediction_instructions(
                    encounter_history=encounter_history,
                    all_positives=all_positives,
                    positives=positives,
                    k_shot=np.random.choice(k_shot_choices),
                    sort_position=True
                )
            multi_task_instructions += instructions

