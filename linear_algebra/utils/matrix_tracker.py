from collections import deque

class Tracker:
    def __init__(self) -> None:
        self.history = deque()

    def r_swap(self, i, j):
        """
        Swaps the position of two rows.
        Ri <-> Rj

        Args:
            i (int): The first row index.
            j (int): The second row index.
        """
        self.history.append({
            "action": "swap",
            "ri": i, # First row index
            "rj": j # Second row index
        })
    
    def r_prod(self, i, j, k):
        """
        Replace a row A for the product of a row B by a skalar plus the row A.
        Ri = Rj * k + Ri

        Args:
            i (int): The row index to change.
            j (int): The row index to multiply.
            k (int): The skalar to multiply.
        """
        self.history.append({
            "action": "prod",
            "ri": i, # Row index to change
            "rj": j, # Row index to add
            "k": k, # Skalar to multiply
        })

    def r_sum(self, i, j):
        """
        Replace a row for the sum of two rows.
        Ri = Ri + Rj

        """
        self.history.append({
            "action": "sum",
            "ri": i, # Row index to change
            "rj": j, # Row index to add
        })
    
    def r_del(self, x):
        """
        Delete rows and added to the bottom.

        Args:
            x (numpy.ndarray): The rows indexes to delete.
        """
        self.history.append({
            "action": "del",
            "list": x, # Rows indexes to delete
        })

    def unqueue(self):
        """
        Unqueue the first action.
        """
        if self.history:
            return self.history.popleft()