"""
Converting between circuit parameters and the EJ, EC the Hamiltonian uses.
Everything returns E/h in GHz.
"""

import numpy as np

e = 1.602176634e-19      # C
h = 6.62607015e-34       # J s
PHI0 = h / (2 * e)       # flux quantum, Wb


def EC_from_capacitance(C_fF):
    """EC = e^2 / 2C, with C the total capacitance of the island."""
    return e**2 / (2 * C_fF * 1e-15) / h / 1e9


def capacitance_from_EC(EC):
    """Inverse of EC_from_capacitance, returns fF."""
    return e**2 / (2 * EC * 1e9 * h) / 1e-15


def EJ_from_critical_current(Ic_nA):
    """EJ = Phi0 Ic / 2 pi."""
    return PHI0 * Ic_nA * 1e-9 / (2 * np.pi) / h / 1e9


def EJ_from_resistance(Rn_kohm, gap_ueV=180.0):
    """Estimate EJ from the junction's normal-state resistance.

    Uses the Ambegaokar-Baratoff relation Ic Rn = pi Delta / 2e. The default
    gap is roughly aluminum's. In practice this is how EJ gets targeted, since
    Rn can be measured at room temperature before the chip is ever cooled.
    """
    Ic = np.pi * gap_ueV * 1e-6 / (2 * Rn_kohm * 1e3)
    return EJ_from_critical_current(Ic * 1e9)


def flux_tuned_EJ(EJ_max, flux, d=0.0):
    """EJ of a SQUID (split) transmon vs external flux in units of Phi0.

    Eq. (2.18): EJ_max |cos(pi flux)| sqrt(1 + d^2 tan^2(pi flux)), where
    EJ_max = EJ1 + EJ2 and d = (EJ2 - EJ1) / (EJ1 + EJ2). Written below as
    sqrt(cos^2 + d^2 sin^2), which is the same thing but stays finite at
    half a flux quantum.
    """
    x = np.pi * np.asarray(flux, dtype=float)
    return EJ_max * np.sqrt(np.cos(x) ** 2 + d**2 * np.sin(x) ** 2)
