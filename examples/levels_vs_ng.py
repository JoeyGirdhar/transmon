"""
Fig. 2 of Koch et al.: the lowest three levels as a function of offset charge,
going from a Cooper pair box (EJ/EC = 1) to a transmon (EJ/EC = 50).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transmon import Transmon

OUT = Path(__file__).resolve().parent.parent / "figures"

ng = np.linspace(-2, 2, 401)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))

for ax, ratio in zip(axes, [1, 5, 10, 50]):
    qubit = Transmon(EJ=ratio, EC=1.0)
    E = np.array([qubit.levels(3, ng=x) for x in ng])
    # same units as the paper: E01 at ng = 1/2, zero at the bottom of level 0
    E = (E - E[:, 0].min()) / qubit.f01
    for m in range(3):
        ax.plot(ng, E[:, m], lw=1.8, label=f"m = {m}")
    for x in (-1.5, -0.5, 0.5, 1.5):
        ax.axvline(x, color="0.8", ls="--", lw=0.8, zorder=0)
    ax.set_title(f"$E_J/E_C$ = {ratio}")
    ax.set_xlabel("$n_g$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(bottom=0)

axes[0].set_ylabel("$E_m / E_{01}$")
axes[-1].legend(frameon=False, fontsize=9, loc="center right")
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "levels_vs_ng.png", dpi=150)
print("saved", OUT / "levels_vs_ng.png")
