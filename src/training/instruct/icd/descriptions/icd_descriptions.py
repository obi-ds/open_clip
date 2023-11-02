"""Map icd codes to textual descriptions"""
import re
import os.path as path
from typing import Optional


class ICDDescription(object):
    """
    Class to convert ICD codes to their textual descriptions
    """

    def __init__(self, icd_source: Optional[str] = None):
        """
        Initialize a mapping from icd code to description

        Args:
            icd_source (Optional[str], defaults to `None`): The file that contains a mapping between codes
            and textual descriptions.
        """
        if icd_source is None:
            icd_source = path.dirname(path.abspath(__file__)) + '/icd_codes.txt'
            self.codes = {line[6:13].strip(): line[76:].strip() for line in open(icd_source)}
        else:
            raise NotImplementedError('Custom icd source files are not supported yet')

    def get_description(self, icd_code: str) -> str:
        """
        Return the textual description for the given code

        Args:
            icd_code (str): The icd code.

        Returns:
            description (str): The textual description of the icd code.
        """
        description = self.codes.get(icd_code.replace('.', ''), '')
        if not re.search(r'\w+', description):
            return icd_code
        return description
