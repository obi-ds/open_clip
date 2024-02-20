"""Map icd codes to textual descriptions"""
import re
import pandas as pd
import os.path as path
from typing import Optional, Mapping, Dict

class CodeDescriptions(object):
    """
    Class to convert codes to their textual descriptions
    """

    def __init__(self, codes: Mapping[str, str]):
        """
        Initialize a mapping from icd code to description

        Args:
            codes (Mapping[str, str]): The object that contains a mapping between codes
            and textual descriptions.
        """
        self.codes = codes

    def get_description(self, code: str) -> str:
        """
        Return the textual description for the given code

        Args:
            code (str): The icd code.

        Returns:
            description (str): The textual description of the icd code.
        """
        raise NotImplementedError('Implement in subclass')


class ICDDescription(CodeDescriptions):
    """
    Class to convert ICD codes to their textual descriptions
    """

    def __init__(
            self,
            source_file: Optional[str] = None
    ):
        """
        Initialize a mapping from icd code to description

        Args:
            source_file (Optional[str], defaults to `None`): The file that contains a mapping between codes
            and textual descriptions.
        """
        if source_file is None:
            source_file = path.dirname(path.abspath(__file__)) + '/icd_codes.txt'
            codes = {line[6:13].strip(): line[76:].strip() for line in open(source_file)}
        else:
            raise NotImplementedError('Custom icd source files are not supported yet')

        super().__init__(codes)

    def get_description(self, code: str) -> str:
        """
        Return the textual description for the given code

        Args:
            code (str): The icd code.

        Returns:
            description (str): The textual description of the icd code.
        """
        description = self.codes.get(code.replace('.', ''), '')
        if not re.search(r'\w+', description):
            print(code)
            return code
        return description


class PHEDescription(CodeDescriptions):
    """
    Class to convert PHE codes to their textual descriptions
    """

    def __init__(self, source_file: Optional[str] = None):
        """
        Initialize a mapping from PHE code to description

        Args:
            source_file (Optional[str], defaults to `None`): The file that contains a mapping between codes
            and textual descriptions.
        """
        if source_file is None:
            source_file = '/mnt/obi0/phi/ehr_projects/bloodcell_clip/data/phecode/phecodeX_info.csv'
            codes = self.get_codes(phe_code_df=pd.read_csv(source_file, encoding='ISO-8859-1'))
        else:
            raise NotImplementedError('Custom icd source files are not supported yet')

        super().__init__(codes)

    @staticmethod
    def get_codes(phe_code_df: pd.DataFrame) -> Dict[str, str]:
        """
        Mapping from code to code description

        Args:
            phe_code_df (pd.DataFrame): The dataframe that contains phe codes
            and their textual descriptions

        Returns:
            (Dict[str, str]): Mapping between code & it's description
        """
        return {row.phecode: row.phecode_string for row in phe_code_df.itertuples()}

    def get_description(self, code: str) -> str:
        """
        Return the textual description for the given code

        Args:
            code (str): The PHE code.

        Returns:
            description (str): The textual description of the PHE code.
        """
        description = self.codes.get(code)
        if not re.search(r'\w+', description):
            raise ValueError('Did not find description of PHE code')
        return description
