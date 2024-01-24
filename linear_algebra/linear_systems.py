import numpy as np
from .echelon_form import ef


def trk_deco(x, logs):
    """
    Decode the logs of Tracker.

    Args:
        x (np.ndarray): Coefficients matrix 
        logs (Tracker): Changes made in matrix 

    Returns:
        s (np.ndarray): Solution vector
    """
    s = x.copy() # Make a copy of the independent terms matrix
    change = logs.unqueue() # Get the first change
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
            tmp = s[rsi] # Save the rows to delete
            s = np.delete(s, rsi, axis=0) # Delete the rows
            s = np.concatenate((s, tmp)) # Concatenate the rows at the bottom

        change = logs.unqueue() # Get the following change

    return s.reshape(-1, 1) # Return the matrix as a column vector



def sol_ls(M, s):
    efM = ef(M)
    return {
        "cf_m": efM["matrix"],  # Matrix of coefficients
        "it_m": trk_deco(s, efM["logs"]),  # Matrix of independent terms
    }
