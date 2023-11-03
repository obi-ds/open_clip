"""Sampling negative icd codes for a patient"""
import numpy as np
import pandas as pd
import os.path as path
from typing import Optional, List

from .icd_convert import ICDConvert

class NegativeICDSampling(object):
    """
    Class to sample negative icd codes for a given patient encounter history
    """

    def __init__(self, code_convert: ICDConvert, source_file: Optional[str] = None):
        """
        Initialize a mapping from icd code to description

        Args:
            code_convert (ICDConvert): Object used to map codes from one form to another
            source_file (Optional[str], defaults to `None`): The file that contains a list of ICD10 codes
        """
        self._code_convert = code_convert
        if source_file is None:
            source_file = path.dirname(path.abspath(__file__)) + '/negatives.txt'
            with open(source_file, 'r') as f:
                self.codes = set(self._code_convert.get_converted_codes(map(str.strip, f.readlines())))
        else:
            raise NotImplementedError('Custom icd source files are not supported yet')

    def get_negatives(
            self,
            positives: pd.Series,
            k: int,
    ) -> List[str]:
        """
        Return a list of icd codes that are not part of the list of positive codes
        Args:
            positives (pd.Series): The complete list of positive icd codes
            k (int): The number of negatives to return

        Returns:
            (List[str]): List containing negative icd codes
        """
        # TODO - maintain a cache of stuff that has been sampled - so we don't sample from that hierarchy  again,
        #  once we have sampled from all distinct hierarchies - reset the cache
        # Remove all the codes present in the patients history from the set of negatives
        negatives = list(self.codes - set(positives))
        return np.random.choice(negatives, k, replace=False)
