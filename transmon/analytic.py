"""
Closed-form results from Koch et al. (2007), used to check the numerics.
"""

from math import factorial

import numpy as np
from scipy.special import mathieu_a, mathieu_b


def mathieu_levels(EJ, EC, ng, count=5):
    """Exact energies from Mathieu characteristic values, eq. (2.2).

    scipy only has characteristic values for integer order, which is exactly
    what the two sweet spots need. Writing phi = 2x turns the Schrodinger
    equation into Mathieu's equation with a = E/EC and q = -EJ/2EC. At ng = 0
    the wavefunction is periodic in phi, which picks out the even orders
    (a_0, b_2, a_2, b_4, ...). At ng = 1/2 the offset charge makes it
    antiperiodic, which picks out the odd ones (a_1, b_1, a_3, ...).

    The sign of q doesn't matter here: flipping it swaps a_r and b_r for odd r
    and leaves even r alone, so the combined, sorted list is the same.
    """
    frac = ng % 1.0
    if np.isclose(frac, 0.0) or np.isclose(frac, 1.0):
        first = 0
    elif np.isclose(frac, 0.5):
        first = 1
    else:
        raise ValueError("Mathieu check only works at ng = 0 or 1/2")

    q = EJ / (2 * EC)
    orders = range(first, first + 2 * count + 2, 2)
    a = [mathieu_a(r, q) for r in orders]
    b = [mathieu_b(r, q) for r in orders if r > 0]
    return EC * np.sort(a + b)[:count]


def asymptotic_levels(EJ, EC, m):
    """Perturbative energies, eq. (2.11). Cosine expanded to fourth order."""
    m = np.asarray(m, dtype=float)
    return (-EJ + np.sqrt(8 * EJ * EC) * (m + 0.5)
            - EC / 12 * (6 * m**2 + 6 * m + 3))


def asymptotic_dispersion(EJ, EC, m):
    """Charge dispersion of level m in the transmon limit, eq. (2.5)."""
    r = np.asarray(EJ / EC, dtype=float)
    return ((-1) ** m * EC * 2 ** (4 * m + 5) / factorial(m)
            * np.sqrt(2 / np.pi) * (r / 2) ** (m / 2 + 3 / 4)
            * np.exp(-np.sqrt(8 * r)))


def asymptotic_n_element(EJ, EC, j):
    """|<j+1|n|j>| for large EJ/EC, eq. (3.4)."""
    return np.sqrt((j + 1) / 2) * (EJ / (8 * EC)) ** 0.25
