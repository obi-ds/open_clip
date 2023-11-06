"""Sampling negative codes for a patient"""
import numpy as np
import pandas as pd
import os.path as path
from typing import Optional, List, Union, Iterable

from .code_convert import ICDConvert, PHEConvert


class NegativeCodeSampling(object):
    """
    Class to sample negative icd codes for a given patient encounter history
    """

    def __init__(self, code_convert: Union[ICDConvert, PHEConvert], raw_codes: Iterable[str]):
        """
        Initialize a mapping from icd code to description

        Args:
            code_convert (ICDConvert): Object used to map codes from one form to another
            raw_codes (Iterable[str]): The raw diagnostic codes
        """
        self._code_convert = code_convert
        self.codes = set(self._code_convert.get_converted_codes(raw_codes))

    def get_codes(
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


class NegativeICDSampling(NegativeCodeSampling):
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
        if source_file is None:
            source_file = path.dirname(path.abspath(__file__)) + '/negatives.txt'
            with open(source_file, 'r') as f:
                raw_codes = map(str.strip, f.readlines())
        else:
            raise NotImplementedError('Custom source file yet to be implemented')
        super().__init__(code_convert=code_convert, raw_codes=raw_codes)


class NegativePHESampling(NegativeCodeSampling):
    """
    Class to sample negative icd codes for a given patient encounter history
    """

    def __init__(self, code_convert: PHEConvert, source_file: Optional[str] = None):
        """
        Initialize a mapping from phe code to description

        Args:
            code_convert (ICDConvert): Object used to map codes from one form to another
            source_file (Optional[str], defaults to `None`): The file that contains a list of ICD10 codes
        """
        if source_file is None:
            source_file = path.dirname(path.abspath(__file__)) + '/../descriptions/phe_codes_flat.csv'
            raw_codes = pd.read_csv(source_file).phecode
        else:
            raise NotImplementedError('Custom source file yet to be implemented')
        super().__init__(code_convert=code_convert, raw_codes=raw_codes)

