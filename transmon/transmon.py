"""
A single transmon, solved in the charge basis.

    H = 4 EC (n - ng)^2 - EJ cos(phi)            Koch et al. (2007), eq. (2.1)

n is the number of Cooper pairs that have tunneled across the junction and phi
is the superconducting phase difference. In the basis of charge states |n> the
first term is diagonal, and cos(phi) = (e^{i phi} + e^{-i phi}) / 2 just moves
one Cooper pair across, so H is a real symmetric tridiagonal matrix.

All energies are E/h in GHz.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal


class Transmon:
    def __init__(self, EJ, EC, ng=0.5, ncut=30):
        self.EJ = float(EJ)
        self.EC = float(EC)
        self.ng = float(ng)
        self.ncut = int(ncut)

    def __repr__(self):
        return (f"Transmon(EJ={self.EJ:g}, EC={self.EC:g}, ng={self.ng:g}, "
                f"ncut={self.ncut})")

    @property
    def ratio(self):
        return self.EJ / self.EC

    def charge_states(self):
        return np.arange(-self.ncut, self.ncut + 1)

    def _bands(self, ng=None):
        ng = self.ng if ng is None else ng
        n = self.charge_states()
        diag = 4 * self.EC * (n - ng) ** 2
        off = np.full(len(n) - 1, -self.EJ / 2)
        return diag, off

    def hamiltonian(self):
        diag, off = self._bands()
        return np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

    def levels(self, count=5, ng=None):
        """Lowest `count` energies in GHz (absolute, not shifted to E0 = 0)."""
        diag, off = self._bands(ng)
        return eigh_tridiagonal(diag, off, eigvals_only=True,
                                select="i", select_range=(0, count - 1))

    def eigensystem(self, count=5):
        diag, off = self._bands()
        vals, vecs = eigh_tridiagonal(diag, off, select="i",
                                      select_range=(0, count - 1))
        # The solver hands back eigenvectors with arbitrary signs. Fix them so
        # that <k|n|k+1> > 0 for every k. Any sign choice is physically fine,
        # this one just makes the matrix elements reproducible.
        n = self.charge_states()
        for k in range(1, count):
            if vecs[:, k - 1] @ (n * vecs[:, k]) < 0:
                vecs[:, k] *= -1
        return vals, vecs

    def n_matrix(self, count=5):
        """<i|n|j> between the lowest `count` eigenstates."""
        _, vecs = self.eigensystem(count)
        return vecs.T @ (self.charge_states()[:, None] * vecs)

    def transition(self, i, j):
        E = self.levels(max(i, j) + 1)
        return E[j] - E[i]

    @property
    def f01(self):
        """Qubit frequency E01/h."""
        return self.transition(0, 1)

    @property
    def anharmonicity(self):
        """alpha = E12 - E01, eq. (2.10). Negative in the transmon regime."""
        E = self.levels(3)
        return (E[2] - E[1]) - (E[1] - E[0])

    def charge_dispersion(self, m):
        """epsilon_m = E_m(ng = 1/2) - E_m(ng = 0), eq. (2.4)."""
        top = self.levels(m + 1, ng=0.5)[m]
        bottom = self.levels(m + 1, ng=0.0)[m]
        return top - bottom
