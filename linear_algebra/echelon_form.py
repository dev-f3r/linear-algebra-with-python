import numpy as np
from math import inf


def is_ef(M):
    """
    Check if a matrix is in echelon form.

    Args:
        M (numpy.ndarray): The matrix.

    Returns:
        bool: True if the matrix is in echelon form, False otherwise.
    """
    # Iterate through all rows except one
    for i in range(M.shape[0] - 1):
        cr = M[i]  # Current row
        nr = M[i + 1]  # Next row

        # If there is a null row it means that the end of the matrix is reached
        # Becuase null rows will be always at the bottom
        if np.all(cr == 0) or np.all(nr == 0):
            return True

        crvi = np.argmax(
            cr != 0
        )  # Get the index of the first non-null val in the current row
        nrvi = np.argmax(
            nr != 0
        )  # Get the index of the first non-null val in the next row

        # The first non-zero element of a row must be at the left of the first non-zero element of the next row.
        if crvi >= nrvi:
            return False

    return True


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


def move_pr_top(X, i, j):
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
    # Swap the the current column with the pivot column
    M[[i, p]] = M[[p, i]]

    return M


def move_nr_bottom(X):
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
    # Delete the null rows
    M = np.delete(M, nri, axis=0)
    # Make a matrix with of null rows
    nrM = np.zeros((len(nri), M.shape[1]))
    # Concatenate the null matrix to the matrix
    M = np.concatenate((M, nrM))

    return M


def ef(X):
    """
    Convert a matrix to echelon form.

    Args:
        X (numpy.ndarray): The matrix.

    Returns:
        numpy.ndarray: The matrix in echelon form.
    """
    # Make a copy of the matrix
    M = np.copy(X)
    # Contains the index of the pivot element
    p = 0
    # Get the number of rows
    r = M.shape[0]
    # Get the number of columns
    c = M.shape[1]
    # Start the iterations with the correct pivot row
    M = move_pr_top(M, 0, 0)
    for i in range(r - 1):
        cr = M[i]  # Current row
        for j in range(i + 1, r):
            nr = M[j]  # Next row
            x, y = cr[p], nr[p]  # Pivot elements of current row and next row

            # If the pivot element of the next row is not zero, the next column must be computed
            if y != 0:
                # If pivot elements are oposite just add the columns
                if x == -y:
                    M[j] = cr + nr
                # If pivot elements are different, find the escalar k that meets x * k + y = 0
                else:
                    k = -y / x
                    M[j] = cr * k + nr

        # * After compute all the rows bellow the pivot row
        # Move the null rows to bottom
        M = move_nr_bottom(M)
        # If the matrix is already in its echelon form return it
        if is_ef(M):
            return M
        # Update the index of the pivot element
        p += 1 if p < c - 1 else 0
        # Find and move the pivot row to the top of the no computed part of the matrix
        M = move_pr_top(M, i + 1, p)

    return M
