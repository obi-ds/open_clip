"""Define various methods to bin the encounter times"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

class FixedDataBins(object):
    """
    Bin data using predefined fixed bins
    """
    # TODO: Implement fixed time bins
    pass


class AgglomerativeDataBins(object):
    """
    Use the AgglomerativeClustering algorithm to bin/cluster time bins
    """

    def __init__(self, distance_threshold=60):
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
