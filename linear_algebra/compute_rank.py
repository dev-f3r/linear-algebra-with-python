import numpy as np
from .echelon_form import ef


def rank(X):
    """
    Calculate the rank of a matrix

    Args:
        X (numpy.ndarray): The matrix.

    Returns:
        int: The rank of the matrix.
    """
    M = np.copy(X)  # Makes a copy of the matrix

    M = ef(M)["matrix"]  # Converts into its echelon form

    # Counts the True values
    return np.count_nonzero(
        np.any(M, axis=1)  # non-null rows as True, null rows as False
    )

def ef_rank(X):
    """
    Calculate the rank of a matrix already in echelon form.

    Args:
        X (numpy.ndarray): The matrix.

    Returns:
        int: The rank of the matrix.
    """

    return np.count_nonzero(
        np.any(X, axis=1)  # non-null rows as True, null rows as False
    )