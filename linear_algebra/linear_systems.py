import numpy as np
from .echelon_form import ef
from .compute_rank import ef_rank


def trk_deco(x, logs):
    """
    Decode the logs of Tracker.

    Args:
        x (np.ndarray): Coefficients matrix
        logs (Tracker): Changes made in matrix

    Returns:
        s (np.ndarray): Solution vector
    """
    s = x.copy()  # Make a copy of the independent terms matrix
    change = logs.unqueue()  # Get the first change
    while change:
        # Swap rows: Ri <-> Rj
        if change.get("action") == "swap" and change["ri"] != change["rj"]:
            i, j = change["ri"], change["rj"]
            s[[i, j]] = s[[j, i]]

        # Replace row: Ri = Rj * k + Ri
        if change.get("action") == "prod":
            i, j, k = change["ri"], change["rj"], change["k"]
            s[i] = s[j] * k + s[i]

        # Replace row: Ri = Ri + Rj
        if change.get("action") == "sum":
            i, j = change["ri"], change["rj"]
            s[i] = s[i] + s[j]

        # Delete rows
        # Contatenate them at the bottom
        if change.get("action") == "del" and change["list"]:
            rsi = change["list"]  # List of rows indexes
            tmp = s[rsi]  # Save the rows to delete
            s = np.delete(s, rsi, axis=0)  # Delete the rows
            s = np.concatenate((s, tmp))  # Concatenate the rows at the bottom

        change = logs.unqueue()  # Get the following change

    return s.reshape(-1)  # Return the matrix as a column vector


def reg_sus(X, xs):
    """
    Apply a regresive sustitution using the coefficients matrix and the independent terms matrix already computed.
    """
    M = X.copy()
    s = xs.copy()
    p = M.shape[1] - 1  # Last val index

    tmpr = np.ones(M.shape[1])  # Temp row to store incognit vals
    tmpr = tmpr.astype(M.dtype)  # Prevent diff data types

    # From last to first row
    for i in range(M.shape[0] - 1, -1, -1):
        cr = M[i]  # Current row

        cr *= tmpr  # Update the current row with incognits

        # From last val until p
        for j in range(M.shape[1] - 1, p - 1, -1):
            cv = cr[j]  # Current val in row

            if cv and cv != 1:
                s[i] += -cv  # Compute sols matrix
                tmpr[p] = s[i]  # Update incognits vals

        p -= 1 if p > 0 else 0  # Update p

    return s


def sol_ls(M, s):
    cm = ef(M)  # Computed matrix in echelon form
    cf_m = cm["matrix"]  # Matrix in echelon form
    it_m = trk_deco(s, cm["logs"])  # Independent terms matrix half computed

    co_m_rank = ef_rank(cf_m)  # Rank of the coeffient matrix
    it_m_rank = np.count_nonzero(it_m)  # Rank of the independent terms matrix

    # Uncompatible
    if co_m_rank != it_m_rank:
        raise Exception(
            f"The sistem is uncompatible, coeffient matrix rank {co_m_rank} and independent terms matrix rank {np.count_nonzero(it_m_rank)}"
        )

    # Compatible undeterminated
    if co_m_rank < cf_m.shape[0]:
        raise Exception(
            f"The sistem is compatible undeterminated, coeffient matrix rank {co_m_rank} and independent terms matrix rank {it_m_rank}"
        )

    # Compatible determinated
    return reg_sus(cf_m, it_m)
