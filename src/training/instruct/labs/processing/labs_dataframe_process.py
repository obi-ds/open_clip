"""Process patient information"""
import pandas as pd
from ...diagnosis.processing import EncounterDataframeProcess


class LabsDataframeProcess(EncounterDataframeProcess):
    """
    Class to process patient information
    """
    def __init__(
            self,
            lab_dataframes: pd.DataFrame,
            patient_id_column: str = 'PatientID',
            contact_date_column: str = 'ResultDTS',
            time_difference_column: str = 'time_difference',
            code_column: str = 'ExternalNM',
            position_column: str = 'position',
    ):
        """
        Processing diagnostic codes, positions and other attributes of the dataframe
        for a given patient and their history.

        Args:
            lab_dataframes (pd.DataFrame): The dataframe that contains encounter history for all patients
            patient_id_column (str, defaults to `PatientID`): The column name for the patient id
            contact_date_column (str, defaults to `ResultDTS`): The column name that contains the time of encounter
            time_difference_column (str, defaults to `time_difference`): The column that will store the time difference
            between a given encounter and all other encounters
            code_column (str, defaults to `ICD10CD`): The column that contains the codes
            position_column (str, defaults to `position`): The column where position values
            (e.g. time difference in days or log days) will be stored
        """
        super().__init__(
            encounter_dataframe=lab_dataframes,
            patient_id_column=patient_id_column,
            contact_date_column=contact_date_column,
            time_difference_column=time_difference_column,
            code_column=code_column,
            position_column=position_column
        )

    def get_encounter_dataframe(self, patient_id):
        """

        Args:
            patient_id:

        Returns:

        """
        index = int(patient_id[-1])
        return self._encounter_dataframe[index]

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
        encounter_dataframe = self.get_encounter_dataframe(patient_id=patient_id)
        return encounter_dataframe.loc[[patient_id]]

    def check_patient_id(self, patient_id: str) -> bool:
        """
        Check if patient id exists in encounter dataframe

        Args:
            patient_id (str): The id of the patient

        Returns:
            (bool): True if patient id exists, false otherwise
        """
        encounter_dataframe = self.get_encounter_dataframe(patient_id=patient_id)
        return patient_id in encounter_dataframe.index
