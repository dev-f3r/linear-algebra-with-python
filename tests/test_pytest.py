import linear_algebra as la
import numpy as np
import pytest

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

def test_beTrue():
    assert np.all(my_sol == np.zeros(my_sol.shape))