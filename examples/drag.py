"""
Why the anharmonicity matters for gates: leakage, and the DRAG fix.

A fast pi pulse has a wide frequency spread. If part of that spread reaches
the 1-2 transition, which sits only |alpha| away from the 0-1 transition,
some population leaks into |2> and is lost from the qubit. For each gate
time, a plain Gaussian pulse and a DRAG pulse are both calibrated (amplitude
and drive frequency, like on a real device) and the remaining X-gate error
is plotted. After calibration nearly all of what's left is leakage.

This only includes coherent errors from the pulse itself, no T1/T2, so the
numbers at long gate times are far below what a real qubit would reach.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transmon import Transmon
from transmon.dynamics import DrivenTransmon, calibrate_pi_pulse, drag_pulse

OUT = Path(__file__).resolve().parent.parent / "figures"

qubit = DrivenTransmon(Transmon(EJ=15, EC=0.3), levels=4)
print(f"f01 = {qubit.f01:.3f} GHz, alpha = {1e3 * qubit.alpha:.0f} MHz")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

T = 10.0
t = np.linspace(0, T, 400)
env = drag_pulse(T, qubit.alpha, beta=1.0)(t)
ax1.plot(t, 1e3 * env.real, lw=1.8, label="in-phase (Gaussian)")
ax1.plot(t, 1e3 * env.imag, lw=1.8, label="quadrature (DRAG)")
ax1.axhline(0, color="0.6", lw=0.8)
ax1.set_xlabel("time (ns)")
ax1.set_ylabel("drive amplitude (MHz)")
ax1.set_title(f"{T:g} ns DRAG pi pulse")
ax1.legend(frameon=False)

gate_times = np.arange(4, 26, 1.0)
for beta, name in [(0.0, "Gaussian"), (1.0, "DRAG")]:
    err = [calibrate_pi_pulse(qubit, T, beta)["error"] for T in gate_times]
    ax2.semilogy(gate_times, err, "o-", ms=4, lw=1.5, label=name)
ax2.set_xlabel("gate time (ns)")
ax2.set_ylabel("X gate error (after calibration)")
ax2.set_title(f"$\\alpha$ = {1e3 * qubit.alpha:.0f} MHz")
ax2.legend(frameon=False)

fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "drag.png", dpi=150)
print("saved", OUT / "drag.png")
