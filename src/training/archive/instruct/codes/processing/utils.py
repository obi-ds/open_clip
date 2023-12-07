import numpy as np
from random import normalvariate


def normal_choice(array, mean=None, stddev=None):
    if mean is None:
        # if mean is not specified, use center of list
        mean = (len(array) - 1) / 2

    if stddev is None:
        # if stddev is not specified, let list be -3 .. +3 standard deviations
        stddev = len(array) / 6

    while True:
        index = int(normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(array):
            return array[index]


def get_log_value(value: float) -> float:
    """
    Get log value of input
    Args:
        value (float): Numerical input

    Returns:
        (float): Log value of input
    """
    if value == 0:
        return value
    else:
        abs_days = np.abs(value)
        return np.sign(value) * np.log2(abs_days)
