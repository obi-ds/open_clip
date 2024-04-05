"""Process patient information"""
import pandas as pd


class DemographicDataframeProcess(object):
    """
    Class to process patient information
    """
    def __init__(
            self,
            demographic_dataframe: pd.DataFrame,
            patient_id_column: str = 'PatientID',
            birth_date_column: str = 'BirthDTS',
            sex_column: str = 'SexDSC',
            race_column: str = 'PatientRaceDSC',
    ):
        """
        Processing diagnostic codes, positions and other attributes of the dataframe
        for a given patient and their history.

        Args:
            demographic_dataframe (pd.DataFrame): The dataframe that contains encounter history for all patients
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            birth_date_column (str, defaults to `ResultDTS`): The column name that contains the time of encounter
            sex_column (str, defaults to `time_difference`): The column that will store the time difference
            between a given encounter and all other encounters
            race_column (str, defaults to `ICD10CD`): The column that contains the codes
        """
        self._demographic_dataframe = demographic_dataframe
        self._patient_id_column = patient_id_column
        self._birth_date_column = birth_date_column
        self._sex_column = sex_column
        self._race_column = race_column
        self._demographic_dataframe = (
            self._demographic_dataframe
            .drop_duplicates(subset=[self._patient_id_column, self._birth_date_column])
            .set_index(self._patient_id_column).sort_index()
        )

    def get_patient_demographics(self, patient_id: str) -> pd.Series:
        """
        For a given patient, return their code encounter history

        Args:
            patient_id (str): Unique patient identifier

        Returns:
            (pd.Series): Series containing information about the patient
        """
        # Extract the patient info from this dataframe - this contains
        # a collection of all codes the patient ever had and the timestamp
        # of their earliest encounters
        return self._demographic_dataframe.loc[patient_id]

    def check_patient_id(self, patient_id: str) -> bool:
        """
        Check if patient id exists in encounter dataframe

        Args:
            patient_id (str): The id of the patient

        Returns:
            (bool): True if patient id exists, false otherwise
        """
        return patient_id in self._demographic_dataframe.index

