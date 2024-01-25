import numpy as np
from .matrix_utils import move_pr_top, move_nr_bottom
from .utils import Tracker


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

def ef(X):
    """
    Convert a matrix to echelon form.

    Args:
        X (numpy.ndarray): The matrix.
    """
    logs = Tracker()
    M = np.copy(X)
    r, c = M.shape

    # i: Pivot row index
    # p: Pivot val in pivot row
    i, p = 0, 0

    # Iterate until M is i echelon form
    while not is_ef(M):
        # Move null rows to bottom
        M = move_nr_bottom(M, logs)
        # Find and move the pivot row to the top of the non computed part of the matrix
        M = move_pr_top(M, i, p, logs)

        cr = M[i] # Current row

        # Iterate through the rest of the rows
        for j in range(i + 1, r):
            nr = M[j] # Next row
            x, y = cr[p], nr[p] # Pivot elements of current and next rows

            # If the pivot element of the next ow is not zero, the next column must be computed
            if y != 0:
                # If pivot elemnts are oposite just add the columns
                if x == -y:
                    M[j] = cr + nr # Change the row
                    logs.r_sum(j, i)
                # If pivot elements are different, find the skalar k that meets x * k + y = 0
                else:
                    k = -y / x
                    M[j] = cr * k + nr # Change the row
                    logs.r_prod(j, i, k)

        # Prevent `i` to reach the last column
        i += 1 if i < r - 2 else 0
        # Prevent `p` to get out of column bound
        p += 1 if p < c - 1 else 0

    return {"matrix": M, "logs": logs}
