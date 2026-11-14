"""
Driving a transmon with microwave pulses.

A drive couples to the qubit through its charge, so in the lab frame

    H(t) = sum_k E_k |k><k| + V(t) n

Keep the lowest few levels, go to a frame rotating at the drive frequency f_d,
and drop the terms that oscillate at ~2 f_d (rotating wave approximation):

    H = sum_k (E_k - k f_d) |k><k|
        + 1/2 sum_k lam_k [ eps(t) |k+1><k| + h.c. ]

eps(t) = Ox(t) + i Oy(t) is the complex pulse envelope in GHz, normalized so
that Ox is the Rabi frequency of the 0-1 transition. lam_k is
<k+1|n|k> / <1|n|0>, close to sqrt(k+1) for a transmon (eq. 3.4). Couplings
that skip a level are left out since they vanish in the transmon limit
(eq. 3.5) and would be dropped by the RWA anyway.

Time is in ns and frequency in GHz, so the propagator is exp(-2 pi i H t).
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

TWO_PI = 2 * np.pi


class DrivenTransmon:
    def __init__(self, transmon, levels=4):
        E, _ = transmon.eigensystem(levels)
        g = transmon.n_matrix(levels)
        self.levels = levels
        self.energies = E - E[0]
        self.f01 = self.energies[1]
        self.alpha = (self.energies[2] - self.energies[1]) - self.f01
        self.lam = np.array([g[k + 1, k] for k in range(levels - 1)]) / g[1, 0]

    def hamiltonian(self, eps, f_drive=None):
        """Rotating-frame H in GHz. `eps` can be a scalar or an array of
        envelope values, in which case this returns a stack of matrices."""
        f_drive = self.f01 if f_drive is None else f_drive
        k = np.arange(self.levels)
        lower = np.diag(self.lam, -1)            # |k+1><k| couplings
        eps = np.asarray(eps, dtype=complex)[..., None, None]
        H0 = np.diag(self.energies - k * f_drive)
        return H0 + 0.5 * (eps * lower + np.conj(eps) * lower.T)

    def propagator(self, envelope, duration, f_drive=None, dt=0.01):
        """Unitary for a shaped pulse. `envelope(t)` returns eps in GHz.

        Piecewise-constant steps with the envelope sampled at each midpoint,
        which is second-order accurate and exactly unitary. All the step
        Hamiltonians get diagonalized in one batched call.
        """
        steps = max(1, int(np.ceil(duration / dt)))
        dt = duration / steps
        t_mid = (np.arange(steps) + 0.5) * dt
        w, v = np.linalg.eigh(self.hamiltonian(envelope(t_mid), f_drive))
        step_U = (v * np.exp(-1j * TWO_PI * w * dt)[:, None, :]) @ v.conj().swapaxes(1, 2)
        U = np.eye(self.levels, dtype=complex)
        for Ui in step_U:
            U = Ui @ U
        return U

    def square_pulse_populations(self, amplitude, times, detuning=0.0):
        """Level populations vs time for a constant drive, starting in |0>.

        H doesn't depend on time here, so diagonalize once and evaluate the
        exponential at every time directly.
        """
        H = self.hamiltonian(amplitude, self.f01 + detuning)
        w, v = eigh(H)
        c0 = v.conj().T[:, 0]                    # <eigenstate|0>
        phases = np.exp(-1j * TWO_PI * np.outer(times, w))
        psi = (phases * c0) @ v.T                # rows are psi(t)
        return np.abs(psi) ** 2


def gaussian(duration, sigma=None, area=0.5):
    """Gaussian envelope on [0, duration], shifted to start and end at zero.

    Scaled so its integral is `area`. The rotation angle is 2 pi times the
    integral of Ox, so area = 0.5 gives a pi pulse.
    Returns (Ox, dOx/dt) as two functions of t.
    """
    sigma = duration / 4 if sigma is None else sigma
    t0 = duration / 2
    floor = np.exp(-t0**2 / (2 * sigma**2))

    def bump(t):
        return np.exp(-(t - t0) ** 2 / (2 * sigma**2))

    ts = np.linspace(0, duration, 4001)
    amp = area / np.trapezoid(bump(ts) - floor, ts)

    def Ox(t):
        return amp * (bump(t) - floor)

    def dOx(t):
        return -amp * (t - t0) / sigma**2 * bump(t)

    return Ox, dOx


def drag_pulse(duration, alpha, beta=1.0, sigma=None, area=0.5):
    """Gaussian pulse with a DRAG correction on the other quadrature.

        Oy(t) = -beta * dOx/dt / (2 pi alpha)

    The 1-2 transition sits alpha away from the drive, so leakage to |2> is
    driven by the part of the pulse spectrum at that offset. With beta = 1
    the derivative term cancels the spectrum there exactly (integrate by
    parts), which removes the leakage to first order. beta = 0 is a plain
    Gaussian. See Motzoi et al., PRL 103, 110501 (2009).
    """
    Ox, dOx = gaussian(duration, sigma, area)

    def envelope(t):
        return Ox(t) - 1j * beta * dOx(t) / (TWO_PI * alpha)

    return envelope


def leakage(U):
    """Population that ends up outside {|0>, |1>}, averaged over both inputs."""
    return 0.5 * (np.sum(np.abs(U[2:, 0]) ** 2) + np.sum(np.abs(U[2:, 1]) ** 2))


def x_gate_error(U):
    """1 - average gate fidelity of an X gate, allowing a free Z correction.

    Uses F = (Tr(M M^dag) + |Tr M|^2) / 6 for the qubit block M of U, which
    counts leakage as error (Pedersen et al., 2007). A Z rotation after the
    gate costs nothing in hardware (it's a frame change), and for an X target
    the best one gives |Tr M| = |U01| + |U10|.
    """
    M = U[:2, :2]
    F = (np.sum(np.abs(M) ** 2) + (abs(M[0, 1]) + abs(M[1, 0])) ** 2) / 6
    return 1 - F


def calibrate_pi_pulse(driven, duration, beta=0.0):
    """Tune the pulse amplitude and drive frequency to minimize X-gate error.

    This is what gets done on a real device: the pulse shape is fixed, then
    amplitude and frequency are scanned until the gate is as good as it gets.
    Here the pulse is simulated instead of measured. Returns a dict with the
    amplitude scale, detuning (GHz), leakage and gate error.
    """
    def run(scale, detuning):
        env = drag_pulse(duration, driven.alpha, beta=beta, area=0.5 * scale)
        return driven.propagator(env, duration, f_drive=driven.f01 + detuning)

    def cost(p):
        return np.log10(x_gate_error(run(p[0], 1e-3 * p[1])) + 1e-15)

    # The error landscape has local minima, so do what you'd do in the lab:
    # a coarse sweep of drive frequency (in MHz) first, then refine.
    starts = [(s, d) for s in (0.95, 1.0, 1.05) for d in range(-60, 61, 10)]
    s0, d0 = min(starts, key=cost)
    simplex = [[s0, d0], [s0 + 0.02, d0], [s0, d0 + 4.0]]
    best = minimize(cost, [s0, d0], method="Nelder-Mead",
                    options=dict(initial_simplex=simplex, xatol=1e-6,
                                 fatol=1e-5, maxiter=600))
    scale, det = best.x[0], 1e-3 * best.x[1]
    U = run(scale, det)
    return dict(scale=scale, detuning=det, leakage=leakage(U),
                error=x_gate_error(U))
