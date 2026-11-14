"""
Tuning the qubit frequency with magnetic flux (split transmon, Sec. II E).

Replacing the single junction with a SQUID loop makes EJ depend on the flux
through the loop, eq. (2.18). With identical junctions (d = 0) EJ goes all the
way to zero at half a flux quantum. Real junctions are never identical, and
the asymmetry d sets how far down the frequency can go.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transmon import Transmon
from transmon.circuit import flux_tuned_EJ

OUT = Path(__file__).resolve().parent.parent / "figures"
EJ_MAX, EC = 20.0, 0.25          # GHz

flux = np.linspace(-1, 1, 401)
fig, ax = plt.subplots(figsize=(6, 4))
for d in [0.0, 0.1, 0.3]:
    f01 = [Transmon(flux_tuned_EJ(EJ_MAX, x, d), EC).f01 for x in flux]
    ax.plot(flux, f01, lw=1.8, label=f"d = {d}")

ax.set_xlabel(r"$\Phi / \Phi_0$")
ax.set_ylabel("$f_{01}$ (GHz)")
ax.set_title(f"$E_{{J,max}}$ = {EJ_MAX:g} GHz, $E_C$ = {EC} GHz")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, title="asymmetry")
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "flux_tuning.png", dpi=150)
print("saved", OUT / "flux_tuning.png")
