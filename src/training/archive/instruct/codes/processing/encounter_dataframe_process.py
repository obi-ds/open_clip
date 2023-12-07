"""Process patient information"""
import pandas as pd
from typing import Optional, Union, Tuple

from .code_convert import ICDConvert, PHEConvert
from .utils import get_log_value


class EncounterDataframeProcess(object):
    """
    Class to process patient information
    """
    def __init__(
            self,
            encounter_dataframe: pd.DataFrame,
            code_convert: Union[ICDConvert, PHEConvert],
            patient_id_column: str = 'PatientID',
            contact_date_column: str = 'ContactDTS',
            time_difference_column: str = 'time_difference',
            code_column: str = 'ICD10CD',
            position_column: str = 'position',
    ):
        """
        Processing diagnostic codes, positions and other attributes of the dataframe
        for a given patient and their history.
        Args:
            encounter_dataframe (pd.DataFrame): The dataframe that contains encounter history for all patients
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            contact_date_column (str, defaults to `ResultDTS`): The column name that contains the time of encounter
            time_difference_column (str, defaults to `time_difference`): The column that will store the time difference
            between a given encounter and all other encounters
            code_column (str, defaults to `ICD10CD`): The column that contains the codes
            position_column (str, defaults to `position`): The column where position values
            (e.g. time difference in days or log days) will be stored
        """
        self._encounter_dataframe = encounter_dataframe
        self._code_convert = code_convert
        self._patient_id_column = patient_id_column
        self._contact_date_column = contact_date_column
        self.time_difference_column = time_difference_column
        self._code_column = code_column
        self._position_column = position_column

    def get_patient_encounter_history(self, patient_id: str) -> pd.DataFrame:
        """
        For a given patient, return their code encounter history
        Args:
            patient_id (str): Unique patient identifier

        Returns:
            (pd.DataFrame): Dataframe containing information about codes in the patient history
        """
        # Extract the patient info from this dataframe - this contains
        # a collection of all codes the patient ever had and the timestamp
        # of their earliest encounters
        return self._encounter_dataframe.loc[[patient_id]]

    @staticmethod
    def get_time_difference(code_timestamps: pd.Series, current_time: str) -> pd.Series:
        """
        Get the time difference between the code encounter and a given time
        Args:
            code_timestamps (pd.Series): Contains the encounter dates of codes
            current_time (str): The time with respect to which we calculate time deltas

        Returns:
            (pd.Series): A series containing the time differences
        """
        return pd.to_datetime(code_timestamps) - pd.to_datetime(current_time)

    @staticmethod
    def get_time_filter(
            time_difference: pd.Series,
            past_time_delta: Optional,
            future_time_delta: Optional
    ) -> pd.Series:
        """
        Return a mask that contains true when the time falls
        within the given range and false otherwise
        Args:
            time_difference (pd.Series): The column containing time deltas
            past_time_delta (): Time delta range in the past
            future_time_delta (): Time delta range for future

        Returns:
            (pd.Series): Mask for filtering based on time deltas
        """
        if past_time_delta is not None:
            min_filter = (time_difference >= -pd.to_timedelta(past_time_delta))
        else:
            min_filter = True
        if future_time_delta is not None:
            max_filter = (time_difference <= pd.to_timedelta(future_time_delta))
        else:
            max_filter = True
        return min_filter & max_filter

    @staticmethod
    def filter_dataframe(dataframe: pd.DataFrame, filter_mask: pd.Series) -> pd.DataFrame:
        """
        Filter dataframe based on the boolean filter
        Args:
            dataframe (pd.DataFrame): Original dataframe
            filter_mask (pd.Series): Boolean filter

        Returns:
            (pd.DataFrame): Filtered dataframe
        """
        return dataframe[filter_mask]

    @staticmethod
    def get_time_difference_normalized(
            time_difference: pd.Series,
            use_log_position: bool,
            time_difference_normalize: int
    ) -> pd.Series:
        """
        Convert the timedelta series into a series that contains the time difference
        in terms of days
        Args:
            time_difference (pd.Series): Series containing time deltas
            use_log_position (bool): Whether to represent the days as raw or log values
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)

        Returns:
            months_difference (pd.Series): Input series represented as difference in terms of days
        """
        if use_log_position:
            time_difference = time_difference.astype(float)
            time_difference = time_difference.apply(get_log_value)
        return time_difference.dt.days / time_difference_normalize

    def map_encounter_codes(self, encounter_history: pd.DataFrame) -> pd.DataFrame:
        """
        Map codes from one format to another and return the dataframe
        with the mapped codes
        Args:
            encounter_history (pd.DataFrame): The input dataframe

        Returns:
            encounter_history (pd.DataFrame): The dataframe with codes mapped
        """
        # Codes can be mapped from leaf to top level - e.g. I48.1 to I48
        encounter_history[self._code_column] = self._code_convert.get_converted_codes(
            codes=encounter_history[self._code_column]
        )
        encounter_history[self._code_column] = encounter_history[self._code_column].astype(str)
        return encounter_history

    def filter_na_codes(self, encounter_history: pd.DataFrame) -> pd.DataFrame:
        """
        If the code column has NA values, replace them accordingly or drop them
        Args:
            encounter_history (pd.DataFrame): The input dataframe

        Returns:
            encounter_history (pd.DataFrame): The dataframe with NA values handled
        """
        encounter_history.dropna(subset=self._code_column, inplace=True)
        return encounter_history

    def filter_encounter_codes(self, encounter_history: pd.DataFrame) -> pd.DataFrame:
        """
        Drop duplicate entries - entries that repeat codes
        Args:
            encounter_history (pd.DataFrame): The input dataframe

        Returns:
            encounter_history (pd.DataFrame): The dataframe with duplicate codes dropped
        """
        # Remove any duplicate entries. When removing duplicates - keep the earliest
        # entry (hence sorted by position)
        encounter_history['temp_int_position'] = encounter_history[self._position_column].astype(int)
        encounter_history.sort_values(by=self._position_column, inplace=True)
        encounter_history.drop_duplicates(subset=[self._code_column, 'temp_int_position'], inplace=True)
        encounter_history.drop(columns=['temp_int_position'], inplace=True)

        return encounter_history

    def filter_encounter_history_time_delta(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: str,
            future_time_delta:str
    ) -> pd.DataFrame:
        """
        Filter encounter history based on the given time range. Keep only those entries
        that occur within the time range
        Args:
            encounter_history (pd.DataFrame): The input dataframe
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time

        Returns:
            encounter_history (pd.DataFrame): Dataframe that contains only those entries within the time frame
        """
        if past_time_delta is not None or future_time_delta is not None:
            # Filter based on time range
            # Keep only the rows within this range
            time_filter = self.get_time_filter(
                time_difference=encounter_history[self.time_difference_column],
                past_time_delta=past_time_delta,
                future_time_delta=future_time_delta
            )
            encounter_history = self.filter_dataframe(
                dataframe=encounter_history,
                filter_mask=time_filter
            )
        return encounter_history

    def get_patient_sequence(
            self,
            patient_id: Union[str, int],
            current_time: str,
            use_log_position: bool,
            time_difference_normalize: int
    ) -> pd.DataFrame:
        """
        For a given patient id, extract the encounter history and apply different
        transformations and operations on the data. Return the transformed dataframe
        that can be used to train the model.
        Args:
            patient_id (Union[str, int]): The unique id of the patient we need to process
            current_time (str): The timestamp of the sample we are processing - used to calculate time deltas
            with respect to other encounters
            use_log_position (bool): Whether to keep time deltas in days or as log value of days
            time_difference_normalize (int, defaults to `30`): Normalize time difference by this value
            (e.g. 30 normalizes it to months)

        Returns:
            encounter_history (pd.DataFrame): The encounter history after applying various functions, transformations
            and sampling
        """
        # Get the code encounter history for a patient
        encounter_history = self.get_patient_encounter_history(patient_id=patient_id)

        # Get time difference with respect to given time
        encounter_history[self.time_difference_column] = self.get_time_difference(
            code_timestamps=encounter_history[self._contact_date_column],
            current_time=current_time
        )

        # Convert time deltas into differences in terms of months
        encounter_history[self._position_column] = self.get_time_difference_normalized(
            time_difference=encounter_history[self.time_difference_column],
            use_log_position=use_log_position,
            time_difference_normalize=time_difference_normalize
        )

        return encounter_history

    def filter_encounter_history_for_task(
            self,
            encounter_history: pd.DataFrame,
            past_time_delta: str,
            future_time_delta: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Given the encounter history, map codes, filter out duplicate codes and filter
        the history based on time range. Keep only those entries
        that occur within the time range
        Args:
            encounter_history (pd.DataFrame): The dataframe containing all encounters
            past_time_delta (str): Filter out encounters beyond this point in time
            future_time_delta (str): Filter out encounters beyond this point in time

        Returns:
            encounter_history (pd.DataFrame): The encounter history after mapping codes, dropping duplicates and
            filtering by time range
            all_positives (pd.Series): The codes present in the entire encounter history of the patient
        """

        encounter_history = self.filter_encounter_codes(
            encounter_history=self.map_encounter_codes(
                encounter_history=self.filter_na_codes(
                    encounter_history=encounter_history
                )
            )
        )

        all_positives = encounter_history[self._code_column]

        encounter_history = self.filter_encounter_history_time_delta(
            encounter_history=encounter_history,
            past_time_delta=past_time_delta,
            future_time_delta=future_time_delta
        )

        return encounter_history, all_positives
