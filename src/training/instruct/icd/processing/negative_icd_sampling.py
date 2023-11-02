"""Sampling negative icd codes for a patient"""
from typing import Optional
import numpy as np
import pandas as pd
import os.path as path
from typing import Optional, List


class NegativeICDSampling(object):
    """
    Class to sample negative icd codes for a given patient encounter history
    # TODO - maintain a cache of stuff that has been sampled - so we don't sample from that hierarchy again, once we have sampled from all distinct hierarchies - reset the cache
    """

    def __init__(self, icd_source: Optional[str] = None):
        """
        Initialize a mapping from icd code to description

        Args:
            icd_source (Optional[str], defaults to `None`): The file that contains a list of ICD10 codes
        """
        if icd_source is None:
            icd_source = path.dirname(path.abspath(__file__)) + '/negatives.txt'
            with open(icd_source, 'r') as f:
                self.codes = set(map(str.strip, f.readlines()))
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
        # Remove all the codes present in the patients history from the set of negatives
        negatives = list(self.codes - set(positives))
        return np.random.choice(negatives, k, replace=False)
