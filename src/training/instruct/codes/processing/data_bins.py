"""Define various methods to bin the encounter times"""
from typing import Union, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


class AgglomerativeDataBins(object):
    """
    Use the AgglomerativeClustering algorithm to bin/cluster time bins
    """

    def __init__(self, distance_threshold):
        # Agglomerate
        self._agglomerative_clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold
        )

    def get_bins(self, data_to_bin: pd.Series) -> np.array:
        """
        Given some piece of data - cluster the data into bin

        Args:
            data_to_bin (pd.Series): The data we want to bin

        Returns:
            (np.array): The resulting cluster bins
        """
        return self._agglomerative_clustering.fit_predict(data_to_bin.values.reshape(-1, 1))


class MonthDataBins(object):
    """
    Use the AgglomerativeClustering algorithm to bin/cluster time bins
    """

    def __init__(self, bins: Optional[Sequence[Tuple[int, int]]] = None):
        if bins is None:
            self._month_bins = [
                (-1, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180),
                (180, 210), (210, 240), (240, 270), (270, 300), (300, 330),
                (330, 360), (360, 390), (390, np.inf)
            ]
        else:
            self._month_bins = bins

    def get_bins(self, data_to_bin: pd.Series) -> np.array:
        """
        Given some piece of data - cluster the data into bin

        Args:
            data_to_bin (pd.Series): The data we want to bin

        Returns:
            (np.array): The resulting cluster bins
        """
        return data_to_bin.apply(self.bin_position)

    def bin_position(self, position: Union[int, float]) -> int:
        """
        Given a position - assign  it a value - depending on which position range it
        belongs to

        Args:
            position (Union[int, float]): A position value

        Returns:
            index (int): A value that assign the position to a defined position range
        """
        # Add 1 to the index - to avoid returning a zero
        for index, position_range in enumerate(self._month_bins):
            if position_range[0] < abs(position) <= position_range[1]:
                if position < 0:
                    return -index - 1
                else:
                    return index + 1
