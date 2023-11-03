"""Map icd codes to textual descriptions"""
import re
import os.path as path
from typing import Optional


class ICDDescription(object):
    """
    Class to convert ICD codes to their textual descriptions
    """

    def __init__(self, source_file: Optional[str] = None):
        """
        Initialize a mapping from icd code to description

        Args:
            source_file (Optional[str], defaults to `None`): The file that contains a mapping between codes
            and textual descriptions.
        """
        if source_file is None:
            source_file = path.dirname(path.abspath(__file__)) + '/icd_codes.txt'
            self.codes = {line[6:13].strip(): line[76:].strip() for line in open(source_file)}
        else:
            raise NotImplementedError('Custom icd source files are not supported yet')

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
            return code
        return description
