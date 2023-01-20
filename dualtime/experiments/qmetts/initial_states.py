import time
import numpy as np


def random_basis(n, zero_magn=False):
    if zero_magn:
        if n % 2 != 0:
            raise ValueError("No!")
        vector = [0] * (n // 2) + [1] * (n // 2)
        vector = np.random.permutation(vector)
    else:
        vector = np.random.randint(0, 2, n)

    bitstring = "".join(map(str, vector))
    return bitstring
