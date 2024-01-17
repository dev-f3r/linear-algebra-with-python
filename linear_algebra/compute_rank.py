import numpy as np
from .echelon_form import ef

def rank(X):
    M = np.copy(X)

    M = ef(M)

    print(np.any(M, axis=1))