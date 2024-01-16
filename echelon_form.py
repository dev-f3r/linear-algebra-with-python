import numpy as np
from math import inf


# def i_non_null(R):
#     """
#     Find the index of the first non-null element in a row.

#     Args:
#         R (numpy.ndarray): The row.

#     Returns:
#         int: The index of the first non-null element.
#     """
#     return np.argmax(R != 0)


def is_echelon_form(M):
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
        crvi (int): The current element index.

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


def move_pivot_row_to_top(M, i, j):
    """
    Move the pivot row to the top.

    Args:
        M (numpy.ndarray): The matrix.
        i (int): The current row index.
        j (int): The current column index.

    Returns:
        numpy.ndarray: The matrix with the pivot row moved to the top.
    """
    p = pri(M, i, j)
    M[[i, p]] = M[[p, i]]
    return M


def move_null_rows_to_bottom(M):
    """
    Move null rows to the bottom.

    Args:
        M (numpy.ndarray): The matrix.

    Returns:
        numpy.ndarray: The matrix with null rows moved to the bottom.
    """
    null_row_indices = np.where(~M.any(axis=1))[0]
    M = np.delete(M, null_row_indices, axis=0)
    null_rows = np.zeros((len(null_row_indices), M.shape[1]))
    M = np.concatenate((M, null_rows))
    return M


def echelon_form(X):
    """
    Convert a matrix to echelon form.

    Args:
        X (numpy.ndarray): The matrix.

    Returns:
        numpy.ndarray: The matrix in echelon form.
    """
    M = np.copy(X)
    p = 0
    r = M.shape[0]
    c = M.shape[1]
    M = move_pivot_row_to_top(M, 0, 0)
    for i in range(r - 1):
        c_row = M[i]
        for j in range(i + 1, r):
            n_row = M[j]
            x, y = c_row[p], n_row[p]
            if y != 0:
                if x == -y:
                    M[j] = c_row + n_row
                else:
                    k = np.linalg.solve(np.array([[x]]), np.array([[-y]]))
                    M[j] = c_row * k[0] + n_row
        M = move_null_rows_to_bottom(M)
        if is_echelon_form(M):
            return M
        p += 1 if p < c - 1 else 0
        M = move_pivot_row_to_top(M, i + 1, p)
    return M


D = np.array(
    [
        [0, 0, 6],
        [-1, -2, -3],
        [1, 2, 3],
        [0, 2, 6],
        [2, 0, -6],
    ]
)

A = np.array(
    [
        [2, -5, 1],
        [1, 4, -2],
        [1, -3, -1],
        [2, 1, -3],
    ]
)

print(echelon_form(D))
print(echelon_form(A))
test = np.array(
    [
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 6.0],
        [0.0, 2.0, 6.0],
        [0.0, -4.0, -12.0],
        [0.0, 0.0, 0.0],
    ]
)
