"""
The tradeoff behind the transmon (Figs. 4a and 5a of Koch et al.).

Left: charge dispersion of the lowest four levels. Solid is the exact
numerical result, dashed is the asymptotic formula, eq. (2.5).
Right: relative anharmonicity E12/E01 - 1, against eq. (2.12).

Raising EJ/EC makes the first one collapse exponentially while the second one
only shrinks like a power law, so there's a wide window where the qubit is
both insensitive to charge noise and still anharmonic enough to control.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transmon import Transmon
from transmon.analytic import asymptotic_dispersion

OUT = Path(__file__).resolve().parent.parent / "figures"
EC = 1.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ratios = np.linspace(1, 120, 300)
for m in range(4):
    exact = np.array([abs(Transmon(r, EC).charge_dispersion(m)) for r in ratios])
    exact[exact < 1e-11] = np.nan      # below this it's just rounding error
    line, = ax1.semilogy(ratios, exact, lw=1.8, label=f"m = {m}")
    big = ratios > 5
    ax1.semilogy(ratios[big], abs(asymptotic_dispersion(ratios[big], EC, m)),
                 "--", color=line.get_color(), lw=1.2)
ax1.set_xlabel("$E_J/E_C$")
ax1.set_ylabel(r"$|\epsilon_m| / E_C$")
ax1.set_title("charge dispersion")
ax1.set_ylim(1e-11, 1e2)
ax1.legend(frameon=False)

ratios = np.geomspace(1, 1000, 300)
alpha_r = []
for r in ratios:
    qubit = Transmon(r, EC)
    alpha_r.append(qubit.anharmonicity / qubit.f01)
ax2.semilogx(ratios, alpha_r, lw=1.8, label="exact")
ax2.semilogx(ratios[ratios > 5], -(8 * ratios[ratios > 5]) ** -0.5, "--",
             lw=1.2, color="C0", label=r"$-(8E_J/E_C)^{-1/2}$")
ax2.axhline(0, color="0.6", lw=0.8)
ax2.set_xlabel("$E_J/E_C$")
ax2.set_ylabel(r"$\alpha_r = \alpha / E_{01}$")
ax2.set_title("relative anharmonicity")
ax2.set_ylim(-0.3, 1.0)
ax2.legend(frameon=False)

fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "tradeoff.png", dpi=150)
print("saved", OUT / "tradeoff.png")
