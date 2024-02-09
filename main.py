import linear_algebra as la
import numpy as np

# * Exercise 8
# * Calculate the rank of each matrix
A = np.array([[1, -1, 1], [0, 1, 0], [1, 1, 0]])
assert la.rank(A) == np.linalg.matrix_rank(A)

B = np.array([[1, 2, 3, 0], [-1, 2, -3, 1], [2, 0, 6, -1]])
assert la.rank(B) == np.linalg.matrix_rank(B)

C = np.array([[-1, 2, -1, 0], [0, -1, 2, 1], [-1, 1, 1, 1], [-1, 0, 3, 2]])
assert la.rank(C) == np.linalg.matrix_rank(C)

D = np.array([[1, 2, 3], [0, 2, 6], [-1, -2, 3], [0, 0, 6], [2, 0, -6]])
assert la.rank(D) == np.linalg.matrix_rank(D)


# * Testing linear systems solver
M = np.array([[1, -2, 3], [0, 1, -2], [0, 0, 1]])
# s = np.array([[3], [-2], [1]])
s = np.array([3, -2, 1])
"""
T = [
    [1,-2,3],
    [0,1,-2],
    [0,0,1]
]
Ts = [
    [3],
    [-2],
    [1]
]

Solution = (0,0,1)
"""
my_sol = la.sol_ls(M, s)
np_sol = np.linalg.solve(M, s)

assert np.all(my_sol == np_sol)