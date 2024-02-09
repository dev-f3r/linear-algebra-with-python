import numpy as np
from math import inf


def pri(M, cri, crvi):
    """
    Find the index of the row with the pivot element.

    Args:
        M (numpy.ndarray): The matrix.
        cri (int): The current row index.
        crvi (int): The current val index.

    Returns:
        int: The index of the row with the pivot element.
    """
    mv = inf  # Minimum val
    mvi = inf  # Index of the minimum val

    # Interate from the actual row until end
    for i in range(cri, M.shape[0]):
        r = M[i]  # Row
        v = abs(r[crvi])  # Pivot element, must be positive

        # If the current row has a 1 as pivot element return the row index
        if v == 1:
            return i

        # Find the minimun
        if v and v < mv:
            mv = v
            mvi = i
    return mvi


def move_pr_top(X, i, j, logs=None):
    """
    Move the pivot row to the top.

    Args:
        X (numpy.ndarray): The matrix.
        i (int): The current row index.
        j (int): The current column index.

    Returns:
        numpy.ndarray: The matrix with the pivot row moved to the top.
    """
    M = np.copy(X)
    # Get the pivot row index
    p = pri(M, i, j)
    # If a pivot row hasn't been found return the matrix.
    if p == inf or p == i:
        return M
    # Swap the the current column with the pivot column
    M[[i, p]] = M[[p, i]]

    if logs:
        logs.r_swap(i, p)

    return M


def move_nr_bottom(X, logs=None):
    """
    Move null rows to the bottom.

    Args:
        X (numpy.ndarray): The matrix.

    Returns:
        numpy.ndarray: The matrix with null rows moved to the bottom.
    """
    M = np.copy(X)

    # Find the indices of the null rows
    nri = np.where(~M.any(axis=1))[0]

    if not nri.size:
        return M

    # Delete the null rows
    M = np.delete(M, nri, axis=0)
    # Make a matrix with of null rows
    nrM = np.zeros((len(nri), M.shape[1]))
    # Concatenate the null matrix to the matrix
    M = np.concatenate((M, nrM))

    if logs:
        logs.r_del(nri)

    return M
