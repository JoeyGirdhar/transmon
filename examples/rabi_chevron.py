"""
Rabi chevron: drive the qubit with a constant microwave tone for a time t at
a detuning from f01, and look at where the population ends up.

On resonance the qubit flips back and forth between |0> and |1> at the Rabi
frequency. Off resonance the oscillation gets faster and shallower, which
gives the chevron shape. The right panel shows the third level: there's a
feature at detuning = alpha/2, where two drive photons together match the
0 -> 2 transition. In a two-level qubit that line wouldn't exist.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transmon import Transmon
from transmon.dynamics import DrivenTransmon

OUT = Path(__file__).resolve().parent.parent / "figures"

qubit = DrivenTransmon(Transmon(EJ=15, EC=0.3), levels=4)
rabi = 0.04                                   # 40 MHz drive
times = np.linspace(0, 200, 401)              # ns
detunings = np.linspace(-0.25, 0.1, 351)      # GHz

pops = np.array([qubit.square_pulse_populations(rabi, times, d)
                 for d in detunings])         # (detuning, time, level)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
extent = [times[0], times[-1], 1e3 * detunings[0], 1e3 * detunings[-1]]
for ax, level in zip(axes, [1, 2]):
    im = ax.imshow(pops[:, :, level], origin="lower", aspect="auto",
                   extent=extent, cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("drive time (ns)")
    ax.set_title(f"population of |{level}>")
    fig.colorbar(im, ax=ax)
axes[0].set_ylabel("drive detuning from $f_{01}$ (MHz)")
axes[1].axhline(1e3 * qubit.alpha / 2, color="w", ls=":", lw=0.8)
axes[1].text(5, 1e3 * qubit.alpha / 2 + 6, r"$\alpha/2$", color="w")

fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "rabi_chevron.png", dpi=150)
print("saved", OUT / "rabi_chevron.png")
